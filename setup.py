from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent

setup(
    name="deltric",
    version="0.4.0",
    description="Delaunay Triangulation Clustering with original-space geometry",
    long_description=(ROOT / "deltric_minimal" / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Tomas Javurek",
    url="https://github.com/kinit-sk/deltric",
    package_dir={"": "deltric_minimal"},
    py_modules=["deltric", "utils_pruning", "utils_component_growth", "plot_seed_component_recomputed_growth", "plot_seed_component_metrics", "hdbscan_comparison"],
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "scipy>=1.10", "scikit-learn>=1.3", "umap-learn>=0.5", "numba>=0.59", "networkx>=3.0", "matplotlib>=3.7"],
)
