from setuptools import find_packages, setup

with open("wilson/_version.py") as f:
    exec(f.read())

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="wilson",
    version=__version__,
    author="Jason Aebischer, David M. Straub",
    author_email="jason.aebischer@tum.de, straub@protonmail.com",
    url="https://github.com/wilson-eft/wilson",
    description="Toolkit for effective field theories beyond the Standard Model",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    package_data={
        "wilson": [
            "run/smeft/tests/data/*.*",
            "data/tests/*.*",
            "data/*.*",
            "wcxf/data/*.yml",
            "wcxf/data/*.yaml",
            "wcxf/data/*.json",
            "wcxf/bases/*.json",
            "wcxf/bases/child/*.json",
        ]
    },
    install_requires=[
        "numpy>=1.16.5",
        "scipy>=1.0",
        "pandas",
        "pylha>=0.2",
        "pyyaml",
        "ckmutil>=1.2.0",
        "rundec>=0.5",
        "voluptuous",
    ],
    python_requires=">=3.6",
    extras_require={"testing": ["nose"]},
    entry_points={
        "console_scripts": [
            "wcxf = wilson.wcxf.cli:wcxf_cli",
            "wcxf2eos = wilson.wcxf.cli:eos",
            "wcxf2dsixtools = wilson.wcxf.cli:wcxf2dsixtools",
            "dsixtools2wcxf = wilson.wcxf.cli:dsixtools2wcxf",
            "wcxf2smeftsim = wilson.wcxf.cli:smeftsim",
        ]
    },
)
