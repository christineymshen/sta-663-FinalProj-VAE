import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VAE-christineymshen",
    version="0.0.1",
    author="Christine Shen, Vidvat Ramachandran",
    author_email="yueming.shen@duke.edu",
    description="A simple implementation of AEVB for MNIST dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/christineymshen/sta-663-FinalProj-VAE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)