import json
import math
from pathlib import Path


def hertz_force(problem):
    indenter = problem["indenter"]
    material = problem["material"]
    stiffness = material["youngs_modulus"] / (1.0 - material["poissons_ratio"] ** 2)
    return 4.0 / 3.0 * stiffness * math.sqrt(indenter["radius"]) * indenter["depth"] ** 1.5


def _force(run):
    return math.sqrt(sum(component * component for component in run["force"]))


def relative_indentation_force_error(files, problem, measurand):
    runs = json.loads(Path(files["observations.json"]).read_text(encoding="utf-8"))["runs"]
    for run in runs:
        if not run["converged"] or not run["finite"]:
            raise ValueError(f"the run at smallest cell {run['smallest']} and scale {run['scale']} did not converge")
        if run["max_penetration"] > run["motion_tolerance"]:
            raise ValueError(f"the run at smallest cell {run['smallest']} and scale {run['scale']} penetrates by {run['max_penetration']}")
    standard = sorted((run for run in runs if run["scale"] == 1.0), key=lambda run: run["smallest"])
    if len(standard) < 2:
        raise ValueError("at least two refinements of the standard block are needed")
    finest, previous = standard[0], standard[1]
    larger = [run for run in runs if run["scale"] > 1.0 and run["smallest"] == finest["smallest"]]
    if not larger:
        raise ValueError("no run on a larger block at the finest cell")
    reference = hertz_force(problem)
    fine = _force(finest)
    tolerance = 1.5 * max(run["motion_tolerance"] for run in runs) / problem["indenter"]["depth"]
    refinement = abs(fine - _force(previous)) / reference
    domain = abs(_force(larger[0]) - fine) / reference
    return {
        "value": abs(fine - reference) / reference,
        "uncertainty": math.sqrt(refinement ** 2 + domain ** 2 + tolerance ** 2),
        "unit": "dimensionless",
    }
