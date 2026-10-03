# Infrared Sensor Simulation and Optimization

Python research software for the design of multispectral infrared microbolometer sensors within the LWI program. The workflow simulates material responses, evaluates their minimum spectral separation, and searches sensor configurations using genetic algorithms and MAP-Elites.

Configuration-driven experiment runners, per-run random seeds and saved design and fitness outputs support reproducible comparisons. MAP-Elites retains different high-scoring designs for review.

This repository contains research and implementation work by Jiahe (LJ) Li at the University of Missouri under the supervision of Dr. Derek Anderson. Simulated response separation is a design objective and a proxy for material discrimination; it does not establish fabricated sensor performance or a global optimum.

## Installation


This project is packaged via standard `pyproject.toml` and requires Python 3.12+.


### Developers


Clone the repo and install in editable mode with dev dependencies (testing, linting, etc.):


```bash
# Clone
git clone https://github.com/LJXXXD/lwi-microbolometer-design.git
cd lwi-microbolometer-design


# Via uv (recommended)
uv sync
uv run pre-commit install   # optional
uv run pytest   # optional


# Or via pip
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install   # optional
pytest   # optional
```


---


## Contact


**Jiahe (LJ) Li** — j.li@missouri.edu — University of Missouri

