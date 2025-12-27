"""
Setup file for Predictive Maintenance package.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="predictive-maintenance",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end MLOps project for Remaining Useful Life (RUL) prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/predictive-maintenance",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pm-train=src.pipelines.training_pipeline:run_training_pipeline",
            "pm-predict=src.components.batch_prediction:run_batch_prediction",
            "pm-api=api.main:app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
