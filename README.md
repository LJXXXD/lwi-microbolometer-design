# LWI Microbolometer Sensor Design and Optimization

This toolkit is developed for the design and optimization of infrared microbolometer sensors within the LWI program. The project focuses on using machine learning (Genetic Algorithms) and physics-based simulations to optimize sensor basis functions for substance detection and identification. Core features include physical response simulation, spectral analysis, and distance metrics (e.g., SAM) for substance discrimination.

This specific repository hosts the research and implementation conducted by Jiahe (LJ) Li at the University of Missouri, under the supervision of Dr. Derek Anderson.

---

## Installation

This project is packaged via standard `pyproject.toml` and requires Python 3.12+.

### Developers

Clone the repo and install in editable mode with dev dependencies (testing, linting, etc.):

```bash
# Clone
git clone [https://github.com/LJXXXD/lwi-microbolometer-design.git](https://github.com/LJXXXD/lwi-microbolometer-design.git)
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
