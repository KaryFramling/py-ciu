# py-ciu

*Explainable Machine Learning through Contextual Importance and Utility*

The *py-ciu* library provides methods to generate post-hoc explanations for
machine learning-based classifiers.

**NOTE: This implementation is the result of a major rewrite made in November 2023. This was made in order to make the use of this py-ciu package similar to the [R CIU](https://github.com/KaryFramling/ciu) and re-written (Nov 2023) [py.ciu.image](https://github.com/KaryFramling/py.ciu.image) packages. Also, the core CIU development is still done in "R" and some of the functionality present in the R version may not be available in this python version.**

The version of py-ciu that has been replaced on ?? December 2023 is available in the branch `VersionUntilNov2023`.

# What is CIU?

**Remark**: It seems like Github Markdown doesn’t show correctly the “{”
and “}” characters in Latex equations, whereas they are shown correctly
in Rstudio. Therefore, in most cases where there is an $i$ shown in
Github, it actually signifies `{i}` and where there is an $I$ it
signifies `{I}`.

CIU is a model-agnostic method for producing outcome explanations of
results of any “black-box” model `y=f(x)`. CIU directly estimates two
elements of explanation by observing the behaviour of the black-box
model (without creating any “surrogate” model `g` of `f(x)`).

**Contextual Importance (CI)** answers the question: ***how much can the
result (or the utility of it) change as a function of feature*** $i$ or a
set of features $\{i\}$ jointly, in the context $x$?

**Contextual Utility (CU)** answers the question: ***how favorable is the
current value*** of feature $i$ (or a set of features $\{i\}$ jointly) for a good
(high-utility) result, in the context $x$?

CI of one feature or a set of features (jointly) $\{i\}$ compared to a
superset of features $\{I\}$ is defined as

$$
\omega_{j,\{i\},\{I\}}(x)=\frac{umax_{j}(x,\{i\})-umin_{j}(x,\{i\})}{umax_{j}(x,\{I\})-umin_{j}(x,\{I\})},  
$$

where $\{i\} \subseteq \{I\}$ and $\{I\} \subseteq \{1,\dots,n\}$. $x$
is the instance/context to be explained and defines the values of input
features that do not belong to $\{i\}$ or $\{I\}$. In practice, CI is
calculated as:

$$
\omega_{j,\{i\},\{I\}}(x)= \frac{ymax_{j,\{i\}}(x)-ymin_{j,\{i\}}(x)}{ ymax_{j,\{I\}}(x)-ymin_{j,\{I\}}(x)}, 
$$

where $ymin_{j}()$ and $ymax_{j}()$ are the minimal and maximal $y_{j}$
values observed for output $j$.

CU is defined as

$$
CU_{j,\{i\}}(x)=\frac{u_{j}(x)-umin_{j,\{i\}}(x)}{umax_{j,\{i\}}(x)-umin_{j,\{i\}}(x)}. 
$$

When $u_{j}(y_{j})=Ay_{j}+b$, this can be written as:

$$
CU_{j,\{i\}}(x)=\left|\frac{ y_{j}(x)-yumin_{j,\{i\}}(x)}{ymax_{j,\{i\}}(x)-ymin_{j,\{i\}}(x)}\right|, 
$$

where $yumin=ymin$ if $A$ is positive and $yumin=ymax$ if $A$ is
negative.

## Usage

First install the `py-ciu` package. The recommended approach is to clone this repository and use it from there. 

The other approach is to install it using `pip install py-ciu` but since the development of CIU is sometimes quite rapid, you should not except that all functionality is available in that version.

A quick overview of the use of py-ciu with different datasets amd ML models is available as a Jupyter notebook [README_notebook.ipynb](README_notebook.ipynb). 

Other notebooks available in the repository are:
- [BostonTests.ipynb](BostonTests.ipynb): Examples of py-ciu use for Boston data set. This notebook provides a good overview of CIU capabilities for a regression task. 
- [TitanicTests.ipynb](TitanicTests.ipynb): Examples of py-ciu use for Titanic data set. This notebook provides a good overview of CIU capabilities for a classification task. It also gives a small example of the use of CIU's "intermediate concepts". 
- [AmesHousingTests.ipynb](AmesHousingTests.ipynb): Examples of py-ciu use for Ames housing data set. This data set has 80 input features and provides a good example of the use of CIU's "intermediate concepts", as well as why they are necessary in order to give "correct" explanations even in the presence of dependencies between features (which is also the case for Titanic). 
- [IrisTests.ipynb](IrisTests.ipynb): Examples of py-ciu use for Iris data set. This notebook includes some "low-level" use of the package that may not be found in the other notebooks. 
- [RunTests.ipynb](RunTests.ipynb): Notebook for running various tests that are found in the `ciu_tests` directory. 

# Related resources

The original R implementation can be found at: <https://github.com/KaryFramling/ciu>

There are also two implementations of CIU for explaining images:

- Python: <https://github.com/KaryFramling/py.ciu.image>
- R: <https://github.com/KaryFramling/ciu.image>

Future work on image explanation will presumably focus on the Python version, due to the extensive use of deep neural networks that tend to be implemented mainly for Python.

## Authors
* [Kary Främling](https://github.com/KaryFramling)
* [Vlad Apopei](https://github.com/vladapopei/)
* [Timotheus Kampik](https://github.com/TimKam/)

The first version of py-ciu was mainly implementd by [Timotheus Kampik](https://github.com/TimKam/) and replaces it. The old code is available in the branch "Historical".

The re-write in 2022 was mainly made by [Vlad Apopei](https://github.com/vladapopei/) and is available in the branch "VersionUntilNov2023".


