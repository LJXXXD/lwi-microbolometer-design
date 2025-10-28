# LWI Microbolometer Sensor Design and Optimization

[![LWI Project](https://img.shields.io/badge/LWI-Project-blue.svg)](https://www.lwi.org/)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the sensor design and optimization toolkit for the LWI program. The primary goal of this project is to develop and apply optimization techniques to design infrared microbolometer sensors for substance detection and identification.

This toolkit was developed as part of the LWI program, with primary development and implementation by Jiahe Li (LJ) under the guidance of Dr. Anderson and collaboration with partner research groups.

## Key Features

* **Sensor Simulation:** Physics-based simulation of microbolometer sensor responses to infrared radiation.
* **Spectral Analysis:** Tools for analyzing emissivity spectra and blackbody radiation calculations.
* **Optimization Algorithms:** Genetic Algorithm (GA) implementation for optimizing sensor basis functions.
* **Distance Metrics:** Spectral Angle Mapper (SAM) and other distance metrics for substance discrimination.
* **Visualization:** Functions for plotting sensor outputs, distance matrices, and optimization results.

## Getting Started

Follow these instructions to set up your local development environment.

### Prerequisites

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
* Git

### Development Installation

**Note:** This installation is for developers working on the toolkit itself. If you're an end user who just wants to use the analysis functions, you can install the package directly with `pip install lwi-microbolometer-design` (once published).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LJXXXD/lwi-microbolometer-design.git
    cd lwi-microbolometer-design
    ```

2.  **Create and activate the Conda environment:**
    This project requires Python 3.12 or newer.
    ```bash
    conda create --name microbolometer-env python=3.12
    conda activate microbolometer-env
    ```

3.  **Install the project in development mode:**
    This installs the package in editable mode (`-e`) with all development dependencies (`[dev]`) needed for testing, linting, and code quality checks.
    ```bash
    pip install -e ".[dev]"
    ```

4.  **Set up pre-commit hooks:**
    This installs automated code quality checks that run before each commit. Run this once after installation.
    ```bash
    pre-commit install
    ```

## Usage

The core functionalities of this package are located in the `src/lwi_microbolometer_design` directory. You can import and use them in your scripts or Jupyter notebooks.

**For detailed usage examples and API documentation, see the `experiments/` directory for Jupyter notebooks demonstrating sensor design and optimization workflows.**

## Development Workflow

This project uses a modern, standardized set of tools to ensure code quality and consistency. All tools are configured in the `pyproject.toml` file.

**Code Quality Tools:**
* **Formatter & Linter:** `ruff` is used for both ultra-fast code formatting and linting.
* **Type Checker:** `mypy` is used for static type checking to prevent bugs.
* **Automation:** `pre-commit` is used to automatically run all checks before any code is committed.

**Development Commands:**
```bash
# Run all code quality checks
pre-commit run --all-files

# Run the test suite
pytest

# Run specific checks
ruff check src/
ruff format src/
mypy src/
```

## Authors

**Primary Developer:** Jiahe Li (LJ) - j.li@missouri.edu - University of Missouri

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
