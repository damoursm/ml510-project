# ML510 Labs

Intro to AI for Enterprise — Polytechnique Montréal

Practical labs for learning to frame business problems, prepare data, build models, evaluate evidence, and make engineering decisions. This repository grows with the course; each available lab has its own setup instructions, notebook sequence, and exercises.

## Available labs

| Lab | Focus | Start here |
|---|---|---|
| S4 — Regression and forecasting | Contextual estimation, model comparison, and forecasting from history | [Setup and lab guide](docs/s4_regression.md) |

S4 is the first available lab. Its notebook assignments are provided through the course Moodle, not GitHub. Download the `s4_regression` folder, extract it if zipped, and copy it into `notebooks/`. See the [notebook placement instructions](notebooks/README.md), then follow the S4 guide for installation and data preparation.

## Repository map

```text
docs/              Setup, sequence, and exercises for each lab
notebooks/         Instructions; place Moodle notebooks here locally
src/ml_lib/        Shared ML teaching library
src/lab_helpers/   Dataset and session-specific helpers
configs/           Lab data and execution settings
data/              Local datasets and explicitly generated data
artifacts/         Saved experiment results and models
scripts/           Environment, data, and execution tools
tests/             Shared-library unit tests
requirements.txt   Shared pinned course dependencies
```

## Shared environment and library

All labs use the course environment defined in `requirements.txt`, which also installs the local packages. Follow the available lab guide to create the environment and select its notebook kernel.

`ml_lib` provides reusable preparation, fitting, evaluation, and artifact handling. `lab_helpers` contains lab-specific data rules. Keep modeling choices visible in the notebooks and inspect detailed results in their linked artifact folders.

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.

## License

Copyright (c) 2026 Houssem Ben Braiek. Original code, notebooks, and course materials in this repository are available under the [MIT License](LICENSE).

Third-party materials retain their own licenses. The UCI Bike Sharing dataset is licensed separately under CC BY 4.0; see the [dataset attribution](docs/s4_regression.md#2-obtain-and-verify-the-data).
