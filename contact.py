import json
import math
from pathlib import Path

from uniaxial import float32_spacing


def reference_slide(problem, time):
    angle = problem["incline"]["angle"]
    friction = problem["incline"]["friction"]
    drive = math.sin(angle) - friction * math.cos(angle)
    if not drive > 0.0:
        raise ValueError(f"the block does not slide: friction {friction} at angle {angle}")
    return 0.5 * problem["gravity"] * time * time * drive


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def relative_slide_distance_error(files, problem, measurand):
    observed = json.loads(Path(files["observations.json"]).read_text(encoding="utf-8"))
    times, com = observed["times"], observed["com"]
    end = problem["duration"]
    if abs(times[-1] - end) > 0.5 * problem["time_step"]:
        raise ValueError(f"trajectory end time {times[-1]} is not the case duration {end}")
    target = reference_slide(problem, end)
    start = com[0]
    slid = _dot([c - s for c, s in zip(com[-1], start)], observed["downhill"])
    window = [_dot([c - s for c, s in zip(point, start)], observed["normal"]) for t, point in zip(times, com) if t >= end - 0.1]
    mean = sum(window) / len(window)
    wobble = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
    largest = max(abs(value) for point in com for value in point)
    absolute = math.sqrt((2.0 * float32_spacing(largest)) ** 2 + wobble ** 2)
    return {"value": abs(slid - target) / target, "uncertainty": absolute / target, "unit": "dimensionless"}


def reference_acceleration(problem):
    return 2.0 * reference_slide(problem, 1.0)


def _trajectory(files, problem):
    observed = json.loads(Path(files["observations.json"]).read_text(encoding="utf-8"))
    times = observed["times"]
    end = problem["duration"]
    if abs(times[-1] - end) > 0.5 * problem["time_step"]:
        raise ValueError(f"trajectory end time {times[-1]} is not the case duration {end}")
    return observed


def _solve3(matrix, vector):
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda r: abs(a[r][column]))
        a[column], a[pivot] = a[pivot], a[column]
        for row in range(3):
            if row != column:
                factor = a[row][column] / a[column][column]
                a[row] = [x - factor * y for x, y in zip(a[row], a[column])]
    return [a[i][3] / a[i][i] for i in range(3)]


def _quadratic_fit(times, values):
    centre = sum(times) / len(times)
    shifted = [t - centre for t in times]
    normal = [[sum(t ** (i + j) for t in shifted) for j in range(3)] for i in range(3)]
    coefficients = _solve3(normal, [sum(v * t ** i for t, v in zip(shifted, values)) for i in range(3)])
    residual = sum((v - sum(c * t ** i for i, c in enumerate(coefficients))) ** 2 for t, v in zip(shifted, values))
    curvature = _solve3(normal, [0.0, 0.0, 1.0])[2]
    return coefficients[2], residual / (len(times) - 3), curvature


def relative_slide_acceleration_error(files, problem, measurand):
    target = reference_acceleration(problem)
    observed = _trajectory(files, problem)
    times, com = observed["times"], observed["com"]
    start = com[0]
    slid = [_dot([c - s for c, s in zip(point, start)], observed["downhill"]) for point in com]

    def fit(begin):
        chosen = [(t, s) for t, s in zip(times, slid) if t >= begin - 1.0e-12]
        return _quadratic_fit([t for t, _ in chosen], [s for _, s in chosen])

    c2, variance, curvature = fit(0.1)
    late, _, _ = fit(0.2)
    largest = max(abs(value) for point in com for value in point)
    acceleration = 2.0 * c2
    standard = 2.0 * math.sqrt(variance * curvature)
    rounding = 2.0 * 2.0 * float32_spacing(largest) * math.sqrt(curvature)
    window = abs(2.0 * late - acceleration)
    absolute = math.sqrt(standard ** 2 + window ** 2 + rounding ** 2)
    return {"value": abs(acceleration - target) / target, "uncertainty": absolute / target, "unit": "dimensionless"}


def relative_normal_drift(files, problem, measurand):
    observed = _trajectory(files, problem)
    com = observed["com"]
    start = com[0]
    element = problem["body"]["height"] / problem["discretization"]["cells"][1]
    drift = max(abs(_dot([c - s for c, s in zip(point, start)], observed["normal"])) for point in com)
    largest = max(abs(value) for point in com for value in point)
    return {"value": drift / element, "uncertainty": 2.0 * float32_spacing(largest) / element, "unit": "dimensionless"}
