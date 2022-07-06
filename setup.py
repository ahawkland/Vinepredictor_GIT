from setuptools import setup, find_packages
from vinepredictor.pathconfig import PathConfig
import src.vinepredictor

DESCRIPTION = 'Machine Learning - Training package'
LONG_DESCRIPTION = 'Machine learning package for training and parameter optimisation'
PACKAGE_NAME = "vinepredictor"

PATHFINDER = PathConfig()


def get_package_version():
    version = src.vinepredictor.VERSION
    print(version)
    return version


# setting up
setup(
    name='vinepredictor',
    version=get_package_version(),
    author='Adam Hertelendi',
    description=DESCRIPTION,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=['numpy', 'pandas', 'scikit-learn'],
    keywords=['python', 'machine learning', 'training', 'grid search'],
    classifiers=[
      "Development Status :: 1 - Planning",
      "Intended Audience :: Data Science",
      "Programming Language :: Python :: 3",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: Microsoft :: Windows"
      ]
)


def main():
    get_package_version()


if __name__ == '__main__':
    main()