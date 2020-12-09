import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vader",
    version="0.0.1",
    description="Deep learning for clustering of multivariate short time series with potentially many missing values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johanndejong/VaDER",
    packages=["vader", "vader.hp_opt"],
    package_dir={'': "tensorflow2"},
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "tensorflow",
    ],
)
