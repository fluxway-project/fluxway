"""
Setup script for Fluxway package.

This file is kept for backward compatibility.
For modern Python projects, pyproject.toml is the preferred configuration.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# Read version from package
version_file = Path(__file__).parent / "fluxway" / "__init__.py"
version = "0.1.0"
if version_file.exists():
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="fluxway",
    version=version,
    description="Fluxway: A Flexible Large-scale Unified eXchange for maximizing Workload Accelerated Yield",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fluxway Contributors",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=["numpy>=1.20.0", "ray>=2.0.0"],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "torch": ["torch>=1.12.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
