"""Run validation, benchmarks, and rebuild figures from saved raw data."""

import os

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
):
    os.environ.setdefault(variable, "1")


def main() -> int:
    from . import figures, plant_study, scalability

    validation = plant_study.run()
    print(validation, flush=True)
    print(scalability.run(), flush=True)
    print(figures.run(), flush=True)
    return int(bool(validation["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
