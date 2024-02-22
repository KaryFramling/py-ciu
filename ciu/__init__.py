"""
This package implements the Contextual Importance and Utility (CIU) method.

Classes: 
    - :class:`ciu.CIU`: The CIU class implements the Contextual Importance and Utility method for Explainable AI. 
    - :class:`ciu.PerturbationMinMaxEstimator.PerturbationMinMaxEstimator`: Class that finds minimal and maximal output values by perturbation of input value(s). This is the default class/method used by `CIU`.

Functions: 
    - `ciu.CIU.contrastive_ciu`: Function for calculating contrastive values from two CIU results. 

Example:
::

    # Example code using the module
    import ciu as ciu
    CIU = ciu.CIU(model.predict_proba, ['Output Name(s)'], data=X_train)
    CIUres = CIU.explain(instance)
    print(CIUres)
"""
from .CIU import CIU

