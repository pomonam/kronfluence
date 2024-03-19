import os

from setuptools import find_packages, setup

version = {}
with open("kronfluence/version.py") as fp:
    exec(fp.read(), version)

__version__ = version["__version__"]

src_dir = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requirements_txt = os.path.join(src_dir, "requirements.txt")
with open("requirements.txt", encoding="utf8") as f:
    required = f.read().splitlines()

with open("dev_requirements.txt", encoding="utf8") as f:
    dev_required = f.read().splitlines()


python_requires = ">=3.9.0"

if __name__ == "__main__":
    setup(
        name="kronfluence",
        version=__version__,
        description="Influence Functions with (Eigenvalue-corrected) Kronecker-factored Approximate Curvature",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        install_requires=required,
        extras_require={
            "dev": dev_required,
        },
        packages=find_packages(),
        keywords=[
            "PyTorch",
            "Training Data Attribution",
            "Influence Functions",
            "KFAC",
            "EKFAC",
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=python_requires,
    )
