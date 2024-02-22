import numpy as np
import pandas as pd
from itertools import product

class PerturbationMinMaxEstimator:
    """
    This class is for abstracting the operation of finding minimal and maximal output value(s)
    for a given instance and given inputs (input indices).

    PerturbationMinMaxEstimator is mainly meant to be used from CIU, not directly! It is the 
    default claas used by CIU for finding minimal and maximal output values but it can be 
    replaced by some other class/object that does it in some (presumably) more efficient way. 
    This can be useful if some model-specific knowledge is available or if there's a reason to do 
    the sampling in a more in-distribution way. 

    The only compulsory method is ``get_minmax_outvals``, which is the method called by CIU with the 
    parameters `ìnstance`` and ``indices``.

    :param predictor: The predictor function to call.
    :param in_minmaxs: DataFrame with as many rows as features and two columns with min and max
        feature values, respectively.
    :param nsamples: How many samples to use.
    :type nsamples: int
    """
    def __init__(self, predictor, in_minmaxs, nsamples):

        self.predictor = predictor
        self.in_minmaxs = in_minmaxs
        self.nsamples = nsamples

    def get_minmax_outvals(self, instance, indices, category_mapping=None):
        """
        Find the minimal and maximal output value(s) that can be obtained by modifying the inputs 
        ``indices`` of the instance ``instance``.

        :param instance: The instance to generate the permuted instances for.
        :param indices: list of indices for which to generate perturbed values.
        
        :return: Two np.arrays with mininmal and maximal output values found for the input or 
            coalition of inputs in ``indices``. 
        """
        samples = self._generate_samples(instance, indices, category_mapping)
        samples_out = self.predictor(samples)
        if samples_out.ndim == 1:
            samples_out = samples_out[:, np.newaxis]
        maxs = np.amax(samples_out,axis=0)
        mins = np.amin(samples_out,axis=0)

        return mins, maxs

    def _generate_samples(self, instance, indices, category_mapping=None):
        """
        Generate a list of instances for estimating CIU.

        :param instance: The instance to generate the permuted instances for.
        :param indices: list of indices for which to generate perturbed values.

        :return: DataFrame with perturbed instances.
        """
        samples_to_do = self.nsamples - 1 # We include the instance in the count

        # Separate indices for numeric features and "category" features. 
        category_indices = []
        fnames = self.in_minmaxs.index
        if category_mapping is None:
            numeric_indices = indices
        else:
            numeric_indices = []
            category_values = []
            for i in indices:
                if fnames[i] in category_mapping:
                    category_indices.append(i)
                    catvals = category_mapping[fnames[i]]
                    # String values have to be converted into numerical.
                    if isinstance(catvals[0], str):
                        catvals = list(range(len(catvals)))
                    category_values.append(catvals)
                else:
                    numeric_indices.append(i)

        # Get categorically perturbed samples.
        catmat = None
        if len(category_indices) > 0:
            catmat = product(*category_values)
            catmat = pd.DataFrame(list(product(*category_values)), columns=fnames[category_indices])
            # If the number of value combinations is bigger than the requested number, then 
            # we need to increase the number accordingly for the numeric features.
            if catmat.shape[0] > samples_to_do:
                samples_to_do = catmat.shape[0]

        # Get numerically perturbed samples.
        # TO-DO: Add rows with all xmin and xmax value combinations for all features.
        numvals = None
        if len(numeric_indices) > 0:
            # First generate all min/max combinations
            mins = np.array(self.in_minmaxs.iloc[numeric_indices,0])
            maxs = np.array(self.in_minmaxs.iloc[numeric_indices,1])
            minmaxgrid = pd.DataFrame(list(product(*self.in_minmaxs.values[numeric_indices,:])))
            nrsamples_to_do = max(0, samples_to_do - minmaxgrid.shape[0])
            # Then fill up the rest with random numbers
            if nrsamples_to_do > 0:
                numvals = np.random.rand(nrsamples_to_do, len(numeric_indices))
                numvals =  mins + (maxs - mins)*numvals
                numvals = pd.concat([minmaxgrid, pd.DataFrame(numvals)], ignore_index=True)
            else: 
                numvals = minmaxgrid
            # This is needed for the case if the number of numeric min/max value combinations 
            # becomes greater than the equested number of samples. 
            if numvals.shape[0] > samples_to_do:
                samples_to_do = numvals.shape[0]

        # Merge numerical and categorical so that the total number of samples is 
        # max(self.nsamples, rows_in_categorical)

        # If no numeric values, then we use only the categorical ones that we have
        if numvals is None:
            samples_to_do = catmat.shape[0]
        samples = pd.concat([instance] * samples_to_do, ignore_index=True)

        # We have categorical values.
        if catmat is not None:
            # Here we may have to expand the categorical values to have same number 
            # of rows as the numerical.
            if numvals is not None and catmat.shape[0] < numvals.shape[0]:
                catmat_nrows = catmat.shape[0]
                numvals_nrows = numvals.shape[0]
                ncopies = int(numvals_nrows/catmat_nrows)
                nrows = (numvals_nrows - catmat_nrows) % catmat_nrows
                l = [pd.concat([catmat]*ncopies), catmat.iloc[np.random.randint(0, catmat_nrows-1, nrows),:]]
                catmat = pd.concat(l)
            samples.iloc[:,category_indices] = catmat
        # We insert numerical columns if we have some
        if numvals is not None:
            samples.iloc[:,numeric_indices] = numvals

        return pd.concat([instance, samples])
    

