import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="FRBID",
  version="0.1",
  long_description=long_description,
  packages=setuptools.find_packages()
)
