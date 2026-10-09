# Solid mechanics pack

A [Terminus](https://github.com/d-adel/terminus) pack of verification cases for solid-mechanics solvers: cases, measurands, schemes and the Python procedures that compute each measurement. MIT licence.

## Contents

| Path | What |
| --- | --- |
| `cases/` | problems for an adapter to realize, with every input |
| `measurands/` | defined quantities; each definition names its reference and how the uncertainty is formed |
| `schemes/` | requirements: a measurand, a limit, the cases and the decision rule |
| `pack.yaml` | maps each `measurand@version` to its procedure |
| `uniaxial.py`, `contact.py`, `indentation.py` | the procedures; standard library only |
| `tests/` | procedure tests against synthetic observations |

## Cases

| Case | Problem type | Reference |
| --- | --- | --- |
| `u1-hex`, `u1-tet` | `uniaxial_tension_unloading`: a bar loaded past yield and unloaded, linear hardening | Bathe, MIT OCW RES.2-002, Topic 17 |
| `c1-incline-0`, `c1-incline-05` | `block_on_incline`: an elastic block sliding down a 45° plane, μ = 0 and 0.5 | Sandia Sierra/SolidMechanics 5.22 Verification Tests Manual §2.14 |
| `c2-hertz` (versions 0 and 1) | `sphere_indentation`: a rigid sphere pressed into an elastic block | Hertz, J. reine angew. Math. 92 (1881) |

## Use

Procedures run from the pack's git commit: `terminus run ... --pack <checkout> --pack-repo <owner/name> --pack-commit <sha>`. Publish measurands first, then cases, then schemes: `terminus publish <file>`.

```
python -m pip install pytest
python -m pytest -q
```
