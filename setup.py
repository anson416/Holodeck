from setuptools import setup

# All package metadata, dependencies, and configuration live in pyproject.toml
# (PEP 621). This shim exists only so legacy `python setup.py ...` invocations
# still work; it must pass no metadata to setup() to avoid conflicting with the
# [project] table.
if __name__ == "__main__":
    setup()
