from setuptools import setup, find_packages

setup(
    name="d_trans",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "dataset-translation = dataset_translation.__main__:main"
        ]
    },
    install_requires=[
        # Add any dependencies your package requires here
    ],
    author="Jasper Schulze Buschhoff",
    author_email="johann.jasper.schulze.buschhoff@iais.fraunhofer.de",
    description="A Python package for dataset translation",
    url="https://github.com/OpenGPTX/dataset-translation",
)
