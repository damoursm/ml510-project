# ML510 Project - Stock Trading Recommendation

Project for the ML510 class, this is a basic stocks trading app that will give recommendations on which stocks to buy for a short term horizon. The recommendations will be produce by an AI model and will be served through an API. 

## Repository map
Based on the labs. 

```text
docs/              Report and documentation
src/ml_lib/        Shared ML teaching library
configs/           Lab data and execution settings
data/              Local datasets and explicitly generated data
artifacts/         Saved experiment results and models
scripts/           Environment, data, and execution tools
tests/             Shared-library unit tests
requirements.txt   Shared pinned course dependencies
```

## Shared environment and library

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests maintain the shared library. Review notebook and lab-helper changes by running the affected notebooks.
