import json
import math
from pathlib import Path

MINIMUM_REFERENCE = 1.0e-4
UNIFORMITY_FACTOR = 10.0


def reference(problem):
    material = problem["material"]
    stress = problem["loading"]["traction"]
    excess = max(stress - material["yield_stress"], 0.0)
    if excess > 0.0 and not material["hardening_modulus"] > 0.0:
        raise ValueError("perfect plasticity under force control has no unique strain past yield")
    plastic = excess / material["hardening_modulus"] if excess > 0.0 else 0.0
    return {"loaded": stress / material["youngs_modulus"] + plastic, "unloaded": plastic}


def _solve3(matrix, vector):
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if rows[pivot][column] == 0.0:
            raise ValueError("rest positions do not span three dimensions")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(column + 1, 3):
            factor = rows[row][column] / rows[column][column]
            for k in range(column, 4):
                rows[row][k] -= factor * rows[column][k]
    solution = [0.0, 0.0, 0.0]
    for row in (2, 1, 0):
        known = sum(rows[row][k] * solution[k] for k in range(row + 1, 3))
        solution[row] = (rows[row][3] - known) / rows[row][row]
    return solution


def affine_fit(rest, deformed):
    count = len(rest)
    mean_rest = [sum(point[d] for point in rest) / count for d in range(3)]
    displacement = [[x[d] - X[d] for d in range(3)] for X, x in zip(rest, deformed)]
    mean_u = [sum(u[d] for u in displacement) / count for d in range(3)]
    centred = [[X[d] - mean_rest[d] for d in range(3)] for X in rest]
    scatter = [[sum(c[i] * c[j] for c in centred) for j in range(3)] for i in range(3)]
    gradient = []
    for component in range(3):
        right = [sum(c[i] * (u[component] - mean_u[component]) for c, u in zip(centred, displacement)) for i in range(3)]
        gradient.append(_solve3(scatter, right))
    squares = 0.0
    for c, u in zip(centred, displacement):
        for component in range(3):
            predicted = mean_u[component] + sum(gradient[component][d] * c[d] for d in range(3))
            squares += (u[component] - predicted) ** 2
    return gradient, math.sqrt(squares / (3 * count))


def float32_spacing(magnitude):
    if magnitude == 0.0:
        return 2.0 ** -149
    return 2.0 ** (math.frexp(magnitude)[1] - 1 - 23)


def _measure(files, problem, stage):
    target = reference(problem)[stage]
    if abs(target) < MINIMUM_REFERENCE:
        raise ValueError(f"reference {stage} strain {target} is below the minimum {MINIMUM_REFERENCE}")
    observed = json.loads(Path(files["observations.json"]).read_text(encoding="utf-8"))
    length = problem["body"]["length"]
    rest, deformed = observed["rest"], observed[stage]
    gradient, rms = affine_fit(rest, deformed)
    largest = max(abs(value) for point in rest + deformed for value in point)
    correction = observed[f"{stage}_solves"][-1]["maxCorrection"]
    terms = [2.0 * rms / length, 2.0 * float32_spacing(largest) / length, 2.0 * correction / length]
    absolute = math.sqrt(sum(term * term for term in terms))
    shear = [0.5 * (gradient[i][j] + gradient[j][i]) for i, j in ((0, 1), (0, 2), (1, 2))]
    lateral_gap = abs(gradient[1][1] - gradient[2][2])
    if max([abs(value) for value in shear] + [lateral_gap]) > UNIFORMITY_FACTOR * absolute:
        raise ValueError(f"the {stage} strain field is not uniaxial: shear {shear}, lateral difference {lateral_gap}, uncertainty {absolute}")
    return {"value": abs(gradient[0][0] - target) / abs(target), "uncertainty": absolute / abs(target), "unit": "dimensionless"}


def relative_strain_error(files, problem, measurand):
    return _measure(files, problem, "loaded")


def relative_permanent_strain_error(files, problem, measurand):
    return _measure(files, problem, "unloaded")


def _transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _inverse(m):
    cofactor = [[m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]
                 - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3] for j in range(3)] for i in range(3)]
    determinant = sum(m[0][j] * cofactor[0][j] for j in range(3))
    if determinant == 0.0:
        raise ValueError("the fitted deformation gradient is singular")
    return [[cofactor[j][i] / determinant for j in range(3)] for i in range(3)]


def polar_stretch(deformation):
    rotation = [row[:] for row in deformation]
    for _ in range(100):
        inverse_t = _transpose(_inverse(rotation))
        updated = [[0.5 * (rotation[i][j] + inverse_t[i][j]) for j in range(3)] for i in range(3)]
        change = max(abs(updated[i][j] - rotation[i][j]) for i in range(3) for j in range(3))
        rotation = updated
        if change < 1e-15:
            break
    stretch = _multiply(_transpose(rotation), deformation)
    return [[0.5 * (stretch[i][j] + stretch[j][i]) for j in range(3)] for i in range(3)]


def residual_bound(rest, deformed, gradient):
    count = len(rest)
    mean_rest = [sum(point[d] for point in rest) / count for d in range(3)]
    mean_u = [sum(x[d] - X[d] for X, x in zip(rest, deformed)) / count for d in range(3)]
    largest = 0.0
    for X, x in zip(rest, deformed):
        for component in range(3):
            predicted = mean_u[component] + sum(gradient[component][d] * (X[d] - mean_rest[d]) for d in range(3))
            largest = max(largest, abs((x[component] - X[component]) - predicted))
    return largest


def _measure_v1(files, problem, stage):
    target = reference(problem)[stage]
    if abs(target) < MINIMUM_REFERENCE:
        raise ValueError(f"reference {stage} strain {target} is below the minimum {MINIMUM_REFERENCE}")
    observed = json.loads(Path(files["observations.json"]).read_text(encoding="utf-8"))
    recent = observed[f"{stage}_solves"][-3:]
    if any(check.get("cgCapped") or not check.get("factored", True) for check in recent):
        raise ValueError(f"the {stage} solve did not finish its linear solve")
    length = problem["body"]["length"]
    rest, deformed = observed["rest"], observed[stage]
    gradient, _ = affine_fit(rest, deformed)
    stretch = polar_stretch([[gradient[i][j] + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)])
    largest = max(abs(value) for point in rest + deformed for value in point)
    correction = max(check["maxCorrection"] for check in recent)
    shear = max(abs(stretch[i][j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    lateral_gap = abs(stretch[1][1] - stretch[2][2])
    terms = [
        3.0 * residual_bound(rest, deformed, gradient) / length,
        3.0 * float32_spacing(largest) / length,
        3.0 * correction / length,
        shear,
        lateral_gap,
    ]
    absolute = math.sqrt(sum(term * term for term in terms))
    return {"value": abs(stretch[0][0] - 1.0 - target) / abs(target), "uncertainty": absolute / abs(target), "unit": "dimensionless"}


def relative_strain_error_v1(files, problem, measurand):
    return _measure_v1(files, problem, "loaded")


def relative_permanent_strain_error_v1(files, problem, measurand):
    return _measure_v1(files, problem, "unloaded")
