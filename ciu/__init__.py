"""
This package implements the Contextual Importance and Utility (CIU) method.

Classes: 
    - :class:`ciu.CIU`: The CIU class implements the Contextual Importance and Utility method for Explainable AI. 
    - :class:`ciu.PerturbationMinMaxEstimator.PerturbationMinMaxEstimator`: Class the finds minimal and maximal output values by perturbation of input value(s). This is the default class/method used by `CIU`.

Example:
::

    # Example code using the module
    import ciu as ciu
    CIU = ciu.CIU(model.predict_proba, ['Output Name(s)'], data=X_train)
    CIUres = CIU.explain(instance)
    print(CIUres)
"""
from .CIU import CIU

