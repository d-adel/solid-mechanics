import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import uniaxial

PROBLEM = {
    "type": "uniaxial_tension_unloading", "units": "SI",
    "body": {"shape": "box", "length": 0.4, "width": 0.05, "height": 0.05},
    "material": {"model": "j2_small_strain_linear_isotropic_hardening", "youngs_modulus": 2.0e8,
                 "poissons_ratio": 0.3, "yield_stress": 2.0e5, "hardening_modulus": 2.0e7, "density": 1000.0},
    "loading": {"type": "equal_and_opposite_end_traction", "axis": "x", "traction": 3.0e5, "then": "unload_to_zero"},
    "supports": "none",
    "discretization": {"element": "linear_hexahedron", "cells": [16, 2, 2]},
}


def grid():
    return [[0.4 * i / 4, 0.05 * j, 0.05 * k] for k in range(2) for j in range(2) for i in range(5)]


def stretched(points, axial, lateral, shear=0.0):
    return [[x + axial * x + shear * y, y + lateral * y, z + lateral * z] for x, y, z in points]


def write(tmp_path, loaded, unloaded, correction=0.0):
    data = {"rest": grid(), "loaded": loaded, "unloaded": unloaded,
            "loaded_solves": [{"maxCorrection": correction}], "unloaded_solves": [{"maxCorrection": correction}]}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"observations.json": str(path)}


def test_reference_matches_hand_calculation():
    expected = uniaxial.reference(PROBLEM)
    assert expected["loaded"] == pytest.approx(1.5e-3 + 5.0e-3, rel=1e-12)
    assert expected["unloaded"] == pytest.approx(5.0e-3, rel=1e-12)


def test_exact_uniform_field_has_no_error(tmp_path):
    files = write(tmp_path, stretched(grid(), 6.5e-3, -0.3 * 1.5e-3 - 2.5e-3), stretched(grid(), 5.0e-3, -2.5e-3))
    for measure in (uniaxial.relative_strain_error, uniaxial.relative_permanent_strain_error):
        result = measure(files, PROBLEM, {})
        assert result["unit"] == "dimensionless"
        assert result["value"] < 1e-9
        assert 0.0 < result["uncertainty"] < 1e-3


def test_wrong_axial_strain_is_measured(tmp_path):
    files = write(tmp_path, stretched(grid(), 6.5e-3 * 1.01, -3.0e-3), stretched(grid(), 5.0e-3, -2.5e-3))
    assert uniaxial.relative_strain_error(files, PROBLEM, {})["value"] == pytest.approx(0.01, rel=1e-6)


def test_convergence_enters_the_uncertainty(tmp_path):
    loaded, unloaded = stretched(grid(), 6.5e-3, -3.0e-3), stretched(grid(), 5.0e-3, -2.5e-3)
    tight = uniaxial.relative_strain_error(write(tmp_path, loaded, unloaded, 0.0), PROBLEM, {})["uncertainty"]
    loose = uniaxial.relative_strain_error(write(tmp_path, loaded, unloaded, 1e-6), PROBLEM, {})["uncertainty"]
    assert loose > tight


def test_sheared_field_is_not_uniaxial(tmp_path):
    files = write(tmp_path, stretched(grid(), 6.5e-3, -3.0e-3, shear=1e-3), stretched(grid(), 5.0e-3, -2.5e-3))
    with pytest.raises(ValueError, match="not uniaxial"):
        uniaxial.relative_strain_error(files, PROBLEM, {})


def test_reference_below_minimum_is_refused(tmp_path):
    elastic = {**PROBLEM, "loading": {**PROBLEM["loading"], "traction": 2.0e5}}
    files = write(tmp_path, stretched(grid(), 1e-3, -3e-4), grid())
    with pytest.raises(ValueError, match="below the minimum"):
        uniaxial.relative_permanent_strain_error(files, elastic, {})


def test_affine_fit_recovers_a_known_gradient():
    gradient, rms = uniaxial.affine_fit(grid(), stretched(grid(), 2e-3, -6e-4, shear=1e-4))
    assert gradient[0][0] == pytest.approx(2e-3, abs=1e-12)
    assert gradient[0][1] == pytest.approx(1e-4, abs=1e-12)
    assert gradient[1][1] == pytest.approx(-6e-4, abs=1e-12)
    assert rms < 1e-15


def rotated(points, angle_x, angle_y):
    import math

    cx, sx, cy, sy = math.cos(angle_x), math.sin(angle_x), math.cos(angle_y), math.sin(angle_y)
    out = []
    for x, y, z in points:
        y, z = cx * y - sx * z, sx * y + cx * z
        x, z = cy * x + sy * z, -sy * x + cy * z
        out.append([x, y, z])
    return out


def write_solves(tmp_path, loaded, unloaded, solves):
    data = {"rest": grid(), "loaded": loaded, "unloaded": unloaded,
            "loaded_solves": solves, "unloaded_solves": solves}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"observations.json": str(path)}


CLEAN = [{"maxCorrection": 0.0, "cgCapped": False, "factored": True}]


def test_v1_reads_strain_without_rigid_rotation_bias(tmp_path):
    loaded = rotated(stretched(grid(), 6.5e-3, -2.95e-3), 0.03, 0.02)
    unloaded = rotated(stretched(grid(), 5.0e-3, -2.5e-3), 0.03, 0.02)
    files = write_solves(tmp_path, loaded, unloaded, CLEAN)
    for measure in (uniaxial.relative_strain_error_v1, uniaxial.relative_permanent_strain_error_v1):
        result = measure(files, PROBLEM, {})
        assert result["value"] < 1e-9
        assert 0.0 < result["uncertainty"] < 1e-3


def test_v1_counts_a_sheared_field_in_the_uncertainty(tmp_path):
    files = write_solves(tmp_path, stretched(grid(), 6.5e-3, -2.95e-3, shear=1e-3), stretched(grid(), 5.0e-3, -2.5e-3), CLEAN)
    result = uniaxial.relative_strain_error_v1(files, PROBLEM, {})
    assert result["uncertainty"] >= 0.5e-3 / 6.5e-3


def test_v1_takes_the_largest_of_the_last_three_corrections(tmp_path):
    loaded, unloaded = stretched(grid(), 6.5e-3, -2.95e-3), stretched(grid(), 5.0e-3, -2.5e-3)
    solves = [{"maxCorrection": c, "cgCapped": False, "factored": True} for c in (1e-3, 1e-6, 1e-6, 1e-9)]
    result = uniaxial.relative_strain_error_v1(write_solves(tmp_path, loaded, unloaded, solves), PROBLEM, {})
    assert result["uncertainty"] >= 3.0 * 1e-6 / 0.4 / 6.5e-3
    assert result["uncertainty"] < 3.0 * 1e-3 / 0.4 / 6.5e-3


def test_v1_refuses_a_capped_linear_solve(tmp_path):
    solves = [{"maxCorrection": 0.0, "cgCapped": True, "factored": True}]
    files = write_solves(tmp_path, stretched(grid(), 6.5e-3, -2.95e-3), stretched(grid(), 5.0e-3, -2.5e-3), solves)
    with pytest.raises(ValueError, match="did not finish"):
        uniaxial.relative_strain_error_v1(files, PROBLEM, {})


def test_perfect_plasticity_is_refused():
    perfect = {**PROBLEM, "material": {**PROBLEM["material"], "hardening_modulus": 0.0}}
    with pytest.raises(ValueError, match="perfect plasticity"):
        uniaxial.reference(perfect)


def test_v1_measures_a_wrong_strain(tmp_path):
    files = write_solves(tmp_path, stretched(grid(), 6.5e-3 * 1.01, -2.95e-3), stretched(grid(), 5.0e-3, -2.5e-3), CLEAN)
    assert uniaxial.relative_strain_error_v1(files, PROBLEM, {})["value"] == pytest.approx(0.01, rel=1e-5)
