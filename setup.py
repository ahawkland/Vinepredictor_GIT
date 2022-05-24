from setuptools import setup, find_packages
VERSION = '0.0.1'
DESCRIPTION = 'Machine Learning - Training package'
LONG_DESCRIPTION = 'Machine learning package for training and parameter optimisation'
# setting up
setup(name='Ml_Training',
      version=VERSION,
      author='Adam Hertelendi',
      description=DESCRIPTION,
      packages=find_packages(where='src'),
      install_requires=['numpy', 'pandas', 'scikit-learn'],
      keywords=['python', 'machine learning', 'training', 'grid search'],
      classifiers=["Development Status :: 1 - Planning",
                   "Intended Audience :: Data Science",
                   "Programming Language :: Python :: 3",
                   "Operating System :: MacOS :: MacOS X",
                   "Operating System :: Microsoft :: Windows"]
      )

