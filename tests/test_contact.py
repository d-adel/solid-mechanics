import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import contact

ANGLE = math.pi / 4
PROBLEM = {"incline": {"angle": ANGLE, "friction": 0.5}, "gravity": 9.81, "duration": 0.5, "time_step": 0.5 / 120}
DOWN = [-math.cos(ANGLE), -math.sin(ANGLE), 0.0]
NORMAL = [-math.sin(ANGLE), math.cos(ANGLE), 0.0]


def trajectory(tmp_path, scale=1.0, wobble=0.0, end=0.5):
    times = [end * i / 120 for i in range(121)]
    reach = 0.5 * 9.81 * (math.sin(ANGLE) - 0.5 * math.cos(ANGLE))
    com = []
    for i, t in enumerate(times):
        s = scale * reach * t * t
        n = wobble * math.sin(40 * t)
        com.append([0.05 + s * DOWN[k] + n * NORMAL[k] for k in range(3)])
    path = tmp_path / "observations.json"
    path.write_text(json.dumps({"times": times, "com": com, "downhill": DOWN, "normal": NORMAL}), encoding="utf-8")
    return {"observations.json": str(path)}


def test_reference_matches_sandia_formula():
    assert contact.reference_slide(PROBLEM, 0.5) == pytest.approx(0.5 * 9.81 * 0.25 * (math.sin(ANGLE) - 0.5 * math.cos(ANGLE)), rel=1e-12)


def test_exact_slide_has_no_error(tmp_path):
    result = contact.relative_slide_distance_error(trajectory(tmp_path), PROBLEM, {})
    assert result["value"] < 1e-12
    assert 0.0 < result["uncertainty"] < 1e-4
    assert result["unit"] == "dimensionless"


def test_slide_error_is_measured(tmp_path):
    assert contact.relative_slide_distance_error(trajectory(tmp_path, scale=1.02), PROBLEM, {})["value"] == pytest.approx(0.02, rel=1e-9)


def test_normal_wobble_enters_the_uncertainty(tmp_path):
    calm = contact.relative_slide_distance_error(trajectory(tmp_path), PROBLEM, {})["uncertainty"]
    shaky = contact.relative_slide_distance_error(trajectory(tmp_path, wobble=1e-4), PROBLEM, {})["uncertainty"]
    assert shaky > calm


def test_no_slide_is_refused(tmp_path):
    stuck = {**PROBLEM, "incline": {"angle": ANGLE, "friction": 1.0}}
    with pytest.raises(ValueError, match="does not slide"):
        contact.relative_slide_distance_error(trajectory(tmp_path), stuck, {})


def test_wrong_end_time_is_refused(tmp_path):
    with pytest.raises(ValueError, match="end time"):
        contact.relative_slide_distance_error(trajectory(tmp_path, end=0.4), PROBLEM, {})


BODY = {**PROBLEM, "body": {"shape": "box", "length": 0.1, "width": 0.1, "height": 0.1}, "discretization": {"element": "linear_hexahedron", "cells": [4, 4, 4]}}


def path_of(tmp_path, slide, normal):
    times = [0.5 * i / 120 for i in range(121)]
    com = [[0.05 + slide(t) * DOWN[k] + normal(t) * NORMAL[k] for k in range(3)] for t in times]
    path = tmp_path / "observations.json"
    path.write_text(json.dumps({"times": times, "com": com, "downhill": DOWN, "normal": NORMAL}), encoding="utf-8")
    return {"observations.json": str(path)}


DRIVE = 9.81 * (math.sin(ANGLE) - 0.5 * math.cos(ANGLE))


def test_acceleration_ignores_a_start_up_velocity_loss(tmp_path):
    files = path_of(tmp_path, lambda t: 0.5 * DRIVE * t * t - 0.003 * t, lambda t: 0.0)
    result = contact.relative_slide_acceleration_error(files, BODY, {})
    assert result["value"] < 1e-9
    assert 0.0 < result["uncertainty"] < 1e-4
    assert result["unit"] == "dimensionless"


def test_acceleration_error_is_measured(tmp_path):
    files = path_of(tmp_path, lambda t: 0.5 * 1.02 * DRIVE * t * t, lambda t: 0.0)
    assert contact.relative_slide_acceleration_error(files, BODY, {})["value"] == pytest.approx(0.02, rel=1e-6)


def test_early_transient_enters_the_acceleration_uncertainty(tmp_path):
    calm = contact.relative_slide_acceleration_error(path_of(tmp_path, lambda t: 0.5 * DRIVE * t * t, lambda t: 0.0), BODY, {})["uncertainty"]
    shaken = contact.relative_slide_acceleration_error(path_of(tmp_path, lambda t: 0.5 * DRIVE * t * t + 1e-4 * math.exp(-20 * t), lambda t: 0.0), BODY, {})["uncertainty"]
    assert shaken > 10 * calm


def test_acceleration_refuses_a_block_that_does_not_slide(tmp_path):
    stuck = {**BODY, "incline": {"angle": ANGLE, "friction": 1.0}}
    with pytest.raises(ValueError, match="does not slide"):
        contact.relative_slide_acceleration_error(path_of(tmp_path, lambda t: 0.0, lambda t: 0.0), stuck, {})


def test_drift_is_the_largest_normal_excursion_per_element(tmp_path):
    files = path_of(tmp_path, lambda t: 0.5 * DRIVE * t * t, lambda t: -1e-4 * t)
    assert contact.relative_normal_drift(files, BODY, {})["value"] == pytest.approx(0.5e-4 / 0.025, rel=1e-6)


def test_no_drift_has_float_resolution_uncertainty_only(tmp_path):
    result = contact.relative_normal_drift(path_of(tmp_path, lambda t: 0.5 * DRIVE * t * t, lambda t: 0.0), BODY, {})
    assert result["value"] < 1e-12
    assert 0.0 < result["uncertainty"] < 1e-4


def test_drift_refuses_the_wrong_end_time(tmp_path):
    with pytest.raises(ValueError, match="end time"):
        contact.relative_normal_drift(path_of(tmp_path, lambda t: 0.0, lambda t: 0.0), {**BODY, "duration": 0.4}, {})
