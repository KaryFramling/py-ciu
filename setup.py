from setuptools import setup, find_packages

# Package metadata
NAME = 'py-ciu'  # Package name
DESCRIPTION = 'Python implementation of the Contextual Importance and Utility (CIU) explainable AI method'
VERSION = '0.1.1'  # Use Semantic Versioning (https://semver.org/)
AUTHOR = 'Vlad Apopei, Kary Främling‚ others'
EMAIL = 'kary.framling@umu.se'
URL = 'https://github.com/KaryFramling/py-ciu'  # Repository URL

# Define your package's dependencies
INSTALL_REQUIRES = [
  'matplotlib',
  'numpy',
  'pandas',
  'scikit-learn',
  'xgboost',
  'scikit_learn',
]

# Long description from README.md
#with open('README.md', 'r') as f:
#    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION = 'Please read the README file'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    license='MIT', 
    classifiers=[
    #    'Development Status :: 3 - Alpha',
    #    'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.7',
    #    'Programming Language :: Python :: 3.8',
    #    'Programming Language :: Python :: 3.9',
    #    'Programming Language :: Python :: 3.10',
    ],
    keywords='Contextual Importance and Utility, CIU, Explainable AI, Explainable Artificial Intelligence',
    #project_urls={
    #    'Source': URL,
    #},
)
