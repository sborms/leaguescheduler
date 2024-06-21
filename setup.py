from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="leaguescheduler",
    version="0.1",
    py_modules=["main"],
    packages=find_packages(),  # finds all packages based on __init__.py file
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "2rr=main:main",
        ],
    },
)
