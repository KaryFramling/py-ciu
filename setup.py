import sys

from setuptools import find_packages, setup

with open('README.md') as f:
    long_desc = f.read()

if sys.version_info < (3, 5):
    print('py-ciu requires at least Python 3.5 to run reliably.')

install_requires = [
    'numpy',
    'pandas'
]

extra_require = {
    'tests': ['pytest']
}

setup(
    name='py-ciu',
    version='0.0.1',
    url='https://github.com/TimKam/py-ciu/',
    license='BSD',
    author='Timotheus Kampik & Sule Anjomshoae',
    author_email='tkampik@cs.umu.se, sulea@cs.umu.se',
    description='Python documentation generator',
    long_description=long_desc,
    long_description_content_type='text/x-md',
    project_urls={
        "Code": "https://github.com/TimKam/py-ciu/",
        "Issue tracker": "https://github.com/TimKam/py-ciu/issues",
    },
    platforms='any',
    packages=find_packages(exclude=['tests']),
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Topic :: Documentation",
    ],
)

