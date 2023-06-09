import pathlib
import setuptools
from importlib.machinery import SourceFileLoader

version = SourceFileLoader("version", "quantile_estimator/version.py").load_module()


def readfile(filename) -> str:
    return pathlib.Path(filename).read_text("utf-8").strip()


setuptools.setup(
    name="quantile-estimator",
    version=version.__version__,
    author="RefaceAI",
    author_email="github-support@reface.ai",
    description=(
        "Python Implementation of Graham Cormode and S. "
        "Muthukrishnan's Effective Computation of Biased "
        "Quantiles over Data Streams in ICDE'05"
    ),
    long_description=readfile("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/RefaceAI/quantile-estimator",
    packages=["quantile_estimator"],
    platforms="Platform Independent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
