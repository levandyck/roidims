from setuptools import setup, find_packages

setup(
    name="roidims",
    version="0.0.1",
    author="Leonard E. van Dyck",
    description="Analysis code",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
