import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import indentation

PROBLEM = {
    "indenter": {"shape": "sphere", "radius": 1.0, "depth": 0.005},
    "block": {"half_width": 1.0, "depth": 1.0},
    "material": {"model": "linear_elastic_isotropic", "youngs_modulus": 2.0e8, "poissons_ratio": 0.3, "density": 1000.0},
}
HERTZ = 4.0 / 3.0 * 2.0e8 / (1.0 - 0.09) * math.sqrt(1.0) * 0.005 ** 1.5


def run(smallest, scale, force, converged=True, penetration=0.0):
    return {"smallest": smallest, "scale": scale, "force": [0.0, -force, 0.0], "converged": converged,
            "max_penetration": penetration, "motion_tolerance": 1.0e-5, "finite": True, "active_nodes": 10}


def observed(tmp_path, runs):
    path = tmp_path / "observations.json"
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return {"observations.json": str(path)}


def ladder(tmp_path, fine=1.0, previous=1.0, big=1.0, **change):
    runs = [run(0.02, 1.0, 1.2 * HERTZ), run(0.01, 1.0, previous * HERTZ), run(0.005, 1.0, fine * HERTZ), run(0.005, 2.0, big * HERTZ)]
    for index, fields in change.items():
        runs[int(index[1:])].update(fields)
    return observed(tmp_path, runs)


def test_reference_is_hertz():
    assert indentation.hertz_force(PROBLEM) == pytest.approx(HERTZ, rel=1e-12)


def test_exact_force_has_only_the_tolerance_uncertainty(tmp_path):
    result = indentation.relative_indentation_force_error(ladder(tmp_path), PROBLEM, {})
    assert result["value"] < 1e-12
    assert result["uncertainty"] == pytest.approx(1.5 * 1.0e-5 / 0.005, rel=1e-9)
    assert result["unit"] == "dimensionless"


def test_error_is_taken_on_the_finest_standard_block(tmp_path):
    assert indentation.relative_indentation_force_error(ladder(tmp_path, fine=1.03, previous=1.03, big=1.03), PROBLEM, {})["value"] == pytest.approx(0.03, rel=1e-9)


def test_last_refinement_and_domain_enter_the_uncertainty(tmp_path):
    result = indentation.relative_indentation_force_error(ladder(tmp_path, previous=1.04, big=0.97), PROBLEM, {})
    assert result["uncertainty"] == pytest.approx(math.sqrt(0.04 ** 2 + 0.03 ** 2 + (1.5 * 1.0e-5 / 0.005) ** 2), rel=1e-9)


def test_an_unconverged_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="converge"):
        indentation.relative_indentation_force_error(ladder(tmp_path, r1={"converged": False}), PROBLEM, {})


def test_a_penetrating_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="penetrat"):
        indentation.relative_indentation_force_error(ladder(tmp_path, r3={"max_penetration": 1.0e-4}), PROBLEM, {})


def test_missing_domain_run_is_refused(tmp_path):
    runs = [run(0.02, 1.0, HERTZ), run(0.01, 1.0, HERTZ), run(0.005, 1.0, HERTZ)]
    with pytest.raises(ValueError, match="larger block"):
        indentation.relative_indentation_force_error(observed(tmp_path, runs), PROBLEM, {})
