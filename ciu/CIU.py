import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from .PerturbationMinMaxEstimator import PerturbationMinMaxEstimator 

class CIU:
    """
    The CIU class implements the Contextual Importance and Utility method for Explainable AI. 

    :param predictor: Model prediction function to be used.
    :param out_names: List of names for the model outputs. This parameter is compulsory because
        it is used for determining how many outputs there are and initializing ``out_minmaxs`` to 0/1 if 
        they are not provided as parameters.
    :type out_names: list
    :param data: DataFrame with the data set to use for inferring min and max input values. Only 
        needed if in_minmaxs is not provided. Default: ``None``.
    :param input_names: list of input column names in ``data``. Default: ``None``.
    :param in_minmaxs: Pandas DataFrame with columns ``min`` and ``max`` and one row per input. Default: ``None``.
    :param out_minmaxs: Pandas DataFrame with columns ``min`` and ``max`` and one row per 
        model output. Default: ``None``.
    :param instance: Instance to be explained, which can be given directly here, which cause an 
        explanation to be calculated straight away, without requiring a call to explain(). Default: ``None``.
    :param output_ind: Default output index to explain. Default: 0. 
    :param vocabulary: Vocabulary to use, defined as a DataFrame. Default: ``None``.
    :param minmax_estimator: Not used yet! For potential abstraction of ymin/ymax estimation in the future, 
        including model-specific methods.  
    """
    def __init__(
        self,
        predictor,
        out_names,
        data=None, 
        input_names=None,
        in_minmaxs=None, 
        out_minmaxs=None, 
        nsamples=100,
        category_mapping=None, 
        neutralCU=0.5, 
        instance=None, 
        output_ind=0,
        vocabulary=None, 
        minmax_estimator=None
    ):
        self.out_names=out_names
        if out_minmaxs is None:
            self.out_minmaxs = pd.DataFrame({'mins': 0, 'maxs': 1}, index=range(len(out_names)))
            self.out_minmaxs.index = out_names
        else:
            self.out_minmaxs = out_minmaxs
        if data is not None:
            input_names = list(data.columns)
            if in_minmaxs is None:
                try:
                    self.in_minmaxs = pd.DataFrame({'mins': data[input_names].min(), 'maxs': data[input_names].max()})
                except:
                    print("Logic Error: You must provide either min_max values or a dataset and input names from which they can be inferred.")
                    raise
        else: 
            self.in_minmaxs=in_minmaxs
            input_names = list(self.in_minmaxs.index)

        self.predictor = predictor
        self.data = data
        self.input_names = input_names
        self.nsamples = nsamples
        self.category_mapping = category_mapping 
        self.neutralCU = neutralCU
        self.instance = instance
        self.output_ind = output_ind
        self.vocabulary = vocabulary
        self.minmax_estimator = minmax_estimator

        # Other instance variables
        self.last_ciu_result = None

    def explain_core(self, coalition_inputs, instance=None, output_inds=None, feature_name=None, nsamples=None, neutralCU = None, 
                     target_inputs=None, out_minmaxs=None, target_concept=None):
        """
        Calculate CIU for a coalition of inputs. This is the "core" CIU method with the actual 
        CIU calculations. All other methods should call this one for doing actual CIU calculations. 

        Coalitions of inputs are used for defining CIU's "intermediate concepts". It signifies that all the 
        inputs in the coalition are perturbed at the same time. 

        :param coalition_inputs: list of input indices. 
        :param instance: Instance to be explained. The default is None. If no instance is passed, then 
            we use the last passed instance by default.
        :param samples: Number of samples to use. Default is ``None``, which means using the default 
            value of the constructor.
        :param neutralCU: Value to use for "neutral CU" in Contextual influence calculation. 
            Default is ``None`` because this parameter is only intended to temporarily override the value 
            given to the constructor. 
        :param target_inputs: list of input indices for "target" concept, i.e. a CIU "intermediate concept". 
            Normally "coalition_inputs" should be a subset of "target_inputs" but that is not a requirement, 
            mathematically taken. Default is None, which signifies that the model outputs (i.e. "all inputs")
            are the targets and the "out_minmaxs" values are used for CI calculation. 
        :param target_cius: DataFrame ... TAKE CARE THAT THIS HAS AS MANY ROWS AS THERE ARE OUTPUTS AND IN THE "RIGHT" 
            ORDER, UNLESS OUTPUT_IND HAS BEEN PROVIDED

        :return: A `list` of DataFrames with CIU results, one for each output of the model. **Remark:** `explain_core()` 
            indeed returns a `list`, which is a difference compared to the two other `explain_` methods! 
        """
        if instance is not None:
            self.instance = instance 
        if self.instance is None:
            raise ValueError("No instance to explain has been given.")
        if nsamples is None:
            nsamples = self.nsamples
        if neutralCU is None:
            neutralCU = self.neutralCU
        outvals = self.predictor(self.instance)
        # We want to make sure that we have a matrix, not an array. 
        if outvals.ndim == 1:
            outvals = outvals[:, np.newaxis]
        nouts = len(self.out_names)

        # Abstraction of MinMaxEstimator comes here, i.e. use default one if none defined. 
        if self.minmax_estimator is None:
            estimator = PerturbationMinMaxEstimator(self.predictor, self.in_minmaxs, nsamples)
        else:
            estimator = self.minmax_estimator
        mins, maxs = estimator.get_minmax_outvals(instance, coalition_inputs, self.category_mapping)
        # If "target_inputs" is given, then we need to get "outmin" and "outmax" values for that 
        # coalition of inputs, rather than for the final outputs.
        if target_inputs is not None:
            target_cius = self.explain_core(target_inputs, instance, nsamples=nsamples, neutralCU=neutralCU)
            all_cius = pd.concat(target_cius)
            outmins = all_cius.loc[:,'ymin']
            outmaxs = all_cius.loc[:,'ymax']
        elif out_minmaxs is not None:
            outmins = out_minmaxs.iloc[:,0]
            outmaxs = out_minmaxs.iloc[:,1]
        else:
            outmins = self.out_minmaxs.iloc[:,0]
            outmaxs = self.out_minmaxs.iloc[:,1]
        cius = []
        if output_inds:
            out_range = [output_inds]
        else:
            out_range = range(nouts)
        for i in out_range:
            outval = outvals[0,i]
            ci = (maxs[i] - mins[i])/(outmaxs[i] - outmins[i]) if (outmaxs[i] - outmins[i]) != 0 else 0
            cu = (outval - mins[i])/(maxs[i] - mins[i]) if (maxs[i] - mins[i]) != 0 else 0
            cinfl = ci*(cu - neutralCU)
            if len(coalition_inputs) == 1:
                fname = self.input_names[coalition_inputs[0]]  
            else:
                fname = "Coalition of %i inputs" % len(coalition_inputs) if feature_name is None else feature_name
            invals = self.instance.iloc[0,coalition_inputs].values
            ciu = pd.DataFrame({'CI': [ci], 'CU': [cu], 'Cinfl': [cinfl], 'outname': [self.out_names[i]], 'outval': [outval],
                                'feature': [fname], 'ymin': [mins[i]], 'ymax': [maxs[i]], 
                                'inputs': [coalition_inputs], 'invals':[invals], 
                                'target_concept': [target_concept], 'target_inputs': [target_inputs]})
            ciu.index.name = 'Feature'
            ciu.index = [[fname]]
            cius.append(ciu)
        return cius

    def explain(self, instance=None, output_ind=None, input_inds=None, nsamples=None, neutralCU=None, 
                vocabulary=None, target_concept=None, target_ciu=None):
        """
        Determines contextual importance and utility for a given instance (set of input/feature values).

        :param instance: Instance to be explained. The default is None.
        :param output_ind: Index of model output to explain. Default is None, in which case it is the value
            of self.output_ind..
        :type output_ind: int
        :param input_inds: list of input indices to include in explanation. Default is None, which 
            signifies "all inputs". 
        :param nsamples: Number of samples to use. Default is ``None``, which means using the default 
            value of the constructor.
        :param neutralCU: Value to use for "neutral CU" in Contextual influence calculation. 
            Default is ``None`` because this parameter is only intended to temporarily override the value 
            given to the constructor.
        """
        # Deal with None parameters.
        if vocabulary is None:
            vocabulary = self.vocabulary
        if output_ind is None:
            output_ind = self.output_ind
        out_minmaxs = None
        if target_concept is None:
            target_inds = None
            if input_inds is None:
                input_inds = list(range(len(self.input_names)))
        else: 
            input_inds = [instance.columns.get_loc(col) for col in self.vocabulary[target_concept]]
            target_inds = [self.input_names.index(value) for value in vocabulary[target_concept]] 
            if target_ciu is not None:
                out_minmaxs = target_ciu.loc[target_concept,['ymin','ymax']]

        # Do the actual work: call explain_core for every input index.
        cius = []
        for i in input_inds:
            ciu = self.explain_core([i], instance, output_inds=output_ind, nsamples=nsamples, neutralCU=neutralCU, 
                                    target_concept=target_concept, target_inputs=target_inds, out_minmaxs=out_minmaxs)[0]
            cius.append(ciu)

        # Memorize last result for direct plotting
        self.last_ciu_result = cius
        return pd.concat(cius)

    def explain_voc(self, instance=None, output_ind=None, input_concepts=None, nsamples=None, neutralCU=None, 
                    vocabulary=None, target_concept=None, target_ciu=None):
        """
        Determines contextual importance and utility for a given instance (set of input/feature values), 
        using the intermediate concept vocabulary.

        :param instance: Instance to be explained. The default is None.
        :param output_ind: Index of model output to explain. Default is 0.
        :param input_inds: list of input indices to include in explanation. Default is None, which 
            signifies "all inputs". 
        :param nsamples: Number of samples to use. Default is ``None``, which means using the default 
            value of the constructor.
        :param neutralCU: Value to use for "neutral CU" in Contextual influence calculation. 
            Default is ``None`` because this parameter is only intended to temporarily override the value 
            given to the constructor. 
        :param vocabulary: Vocabulary to use instead of the one (presumably) given to the constructor. Default: ``None``.
        """

        # Deal with None parameters.
        if vocabulary is None:
            vocabulary = self.vocabulary
        if output_ind is None:
            output_ind = self.output_ind
        out_minmaxs = None
        if target_concept is None:
            target_inds = None
        else: 
            target_inds = [self.input_names.index(value) for value in vocabulary[target_concept]] 
            if target_ciu is not None:
                out_minmaxs = target_ciu.loc[target_concept,['ymin','ymax']]
        if input_concepts is None:
            input_concepts = list(vocabulary.keys())

        # Do the actual work: call explain_core for every input index.
        cius = []
        for ic in input_concepts:
            inds = [self.input_names.index(value) for value in vocabulary[ic]] 
            ciu = self.explain_core(inds, instance, output_inds=output_ind, nsamples=nsamples, neutralCU=neutralCU, feature_name=ic, 
                                    target_concept=target_concept, target_inputs=target_inds,  out_minmaxs=out_minmaxs)[0]
            cius.append(ciu)

        # Memorize last result for direct plotting
        self.last_ciu_result = cius
        return pd.concat(cius)
    
    def explain_all(self, data=None, output_ind=None, input_inds=None, nsamples=None, neutralCU=None, 
                vocabulary=None, target_concept=None, target_ciu=None, do_norm_invals=False):
        """
        Do CIU for all instances in `data`. 

        :param: do_norm_invals: Should a column with normalized input values be produced or not? This 
        can only be done for "basic" features, not for coalitions of features (intermediate concepts) at 
        least for the moment. It is useful to provide normalized input values for getting more 
        meaningful beeswarm plots, for instance. Default: False.

        :return: DataFrame with CIU results if all instances concatenated.
        """
        # Deal with None parameters.
        if data is None:
            if self.data is not None:
                data = self.data
            else:
                raise ValueError("No data provided.")
        
        # Get values that are needed for normalizing input values.
        if do_norm_invals:
            minmaxrows = self.in_minmaxs.loc[list(data.columns),:]
            mins = np.array(minmaxrows.iloc[:,0])
            maxs = np.array(minmaxrows.iloc[:,1])
            ranges = maxs - mins

        # Go through all the instances (rows) in the data
        ciu_res = []
        for i in range(len(data)):
            instance = data.iloc[[i]]
            ciu = self.explain(instance, output_ind=output_ind, input_inds=input_inds, 
                                        nsamples=nsamples, neutralCU=neutralCU, vocabulary=vocabulary, 
                                        target_concept=target_concept, target_ciu=target_ciu)
            row_name = list(instance.index)[0]
            ciu['instance_name'] = [str(row_name)]*len(ciu)
            if do_norm_invals:
                ciu['norm_invals'] = ciu['invals']
                ciu = ciu.explode('norm_invals',) # Get rid of list
                ninvals = np.array(ciu.loc[:,'norm_invals'])
                ciu.loc[:,'norm_invals'] = (ninvals - mins)/ranges
            ciu_res.append(ciu)
        return pd.concat(ciu_res, ignore_index=True)

#============================================================================================
# Plotting and textual explanation functions here, which tend to become very long.
#============================================================================================

    # Input/output plot, with possibility to illustrate CIU.
    def plot_input_output(self, instance=None, ind_input=0, ind_outputs=0, in_min_max_limits=None,
                         n_points=40, main=None, xlab="x", ylab="y", ylim=0, figsize=(6, 4),
                         illustrate_CIU=False, legend_location=0, neutral_CU=0.5, 
                         CIU_illustration_colours=None):
        #c("red", "orange", "green", "blue")
        """
        Plot model output(s) value(s) as a function on one input. 

        :param: ind_outputs: Integer value, list of integers or None. If None then all outputs are plotted. 
            Default: 0. 
        :param ylim: Value limits for y-axis. Can be zero, actual limits or None. Zero signifies that the known 
            min/max values for the output will be used. ``None`` signifies that no limits are defined and are 
            auto-determined by ``plt.plot``. If actual limits are given, they are passed to ``plt.ylim`` as such. 
            Default: zero. 
        """

        # Deal with None parameters and other parameter value arrangements.
        if instance is None:
            instance = self.instance
        if ind_outputs is None:
            ind_outputs = list(range(len(self.out_names)))
        elif type(ind_outputs) is not list:
            ind_outputs = [ind_outputs]
 
        # Check is it's a numeric or categorical input.
        fname = self.input_names[ind_input]
        if self.category_mapping is None or fname not in self.category_mapping:
            input_type = 'N'
        else:
            input_type = 'C'

        # First deal with "numeric" possibility.
        if input_type == 'N':
            if in_min_max_limits is None:
                in_min_max_limits = self.in_minmaxs.iloc[ind_input,:]
            in_min = in_min_max_limits[0]
            in_max = in_min_max_limits[1]
            interv = (in_max - in_min)/n_points
            x = np.arange(in_min, in_max, interv)
            m = np.tile(instance, (n_points, 1))
        else:
            x = self.category_mapping[fname]
            xlabels = x
            if isinstance(x[0], str):
                x = list(range(len(x)))
            m = np.tile(instance, (len(x), 1))

        m[:,ind_input] = x
        y = self.predictor(pd.DataFrame(m, columns=self.input_names))
        if y.ndim == 1:
            y = y[:,np.newaxis]
        outvals = self.predictor(instance)
        if outvals.ndim == 1:
            outvals = outvals[:,np.newaxis]
 
        # Do actual plotting. If None is given, then we plot all outputs
        plt_out_names = [self.out_names[i] for i in ind_outputs] if len(ind_outputs) > 1 else self.out_names[ind_outputs[0]]
        fig, ax = plt.subplots(figsize=figsize)

        if input_type == 'N':
            plt.plot(x, y[:, ind_outputs], label=plt_out_names)
            # circle_radius = 0.5
            # plt.scatter(instance.iloc[0,ind_input], cu_val, color='red', marker='o', s=circle_radius**2 * 100)
        else: 
            plt.bar(xlabels, y[:, ind_outputs[0]], label=plt_out_names)
        
        # Plot current value(s) as dot(s)
        repx = np.repeat(instance.iloc[0,ind_input], len(ind_outputs))
        plt.scatter(repx, outvals[0,ind_outputs], color='red', marker='o', label='out') # This radius seems OK

        # Decide on y-limits
        if ylim == 0:
            ylim = (np.amin(self.out_minmaxs.iloc[[0],0][0]), np.amax(self.out_minmaxs.iloc[[0],1][0]))
            plt.ylim(ylim)
        elif ylim is not None:
            plt.ylim(ylim)
 
        if illustrate_CIU:
            y_min = np.amin(y[:, ind_outputs])
            plt.axhline(y=y_min, color='red', linestyle='--', label='ymin')
            #plt.text(max(x), y_min, 'ymin', verticalalignment='top', horizontalalignment='right', color='red')
            y_max = np.amax(y[:, ind_outputs])
            plt.axhline(y=y_max, color='green', linestyle='--', label='ymax')
            #plt.text(max(x), y_max, 'ymax', verticalalignment='bottom', horizontalalignment='right', color='green')

        # Legend?
        if legend_location is not None:
            plt.legend(loc=legend_location)

        # Add titles.
        if main is None:
            main = 'Output value as a function of feature value'
        plt.title(main)
        plt.xlabel(self.input_names[ind_input])
        if len(ind_outputs) == 1:
            plt.ylabel(self.out_names[ind_outputs[0]])
        else:
            plt.ylabel('Output values')

    def plot_ciu(self, ciu_result, plot_mode='default', CImax=1.0, 
                sort='CI', main=None, color_blind=None, figsize=(6, 4),
                color_fill_ci='#7fffd44d', color_edge_ci='#66CDAA',
                color_fill_cu="#006400cc", color_edge_cu="#006400"):

        """
        :param ciu_result: CIU result DataFrame as returned by one of the "explain..." methods. 
        :param str plot_mode: defines the type plot to use between 'default', 'overlap' and 'combined'.
        :param CImax: Limit CI axis to the given value. Default is 1
        :param str sort: defines the order of the plot bars by the 'CI' (default), 'CU' values or unsorted if None;
        :param str color_blind: defines accessible color maps to use for the plots, such as 'protanopia',
            'deuteranopia' and 'tritanopia'.
        :param str color_edge_cu: defines the hex or named color for the CU edge in the overlap plot mode.
        :param str color_fill_cu: defines the hex or named color for the CU fill in the overlap plot mode.
        :param str color_edge_ci: defines the hex or named color for the CI edge in the overlap plot mode.
        :param str color_fill_ci: defines the hex or named color for the CI fill in the overlap plot mode.
        """

        feature_names = ciu_result.feature
        ci = ciu_result.CI
        cu = ciu_result.CU
        nfeatures = len(feature_names)

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(nfeatures)

        if sort in ['CI', 'influence']:
            ci, cu, feature_names = (list(t) for t in zip(*sorted(zip(ci, cu, feature_names))))
        elif sort == 'CU':
            cu, ci, feature_names = (list(t) for t in zip(*sorted(zip(cu, ci, feature_names))))

        my_norm = colors.Normalize(vmin=0, vmax=1)
        nodes = [0.0, 0.5, 1.0]

        # Take care of available color palettes.
        if color_blind is None:
             colours = ["red", "yellow", "green"]
        elif color_blind == 'protanopia':
            colours = ["gray", "yellow", "blue"]
        elif color_blind == 'deuteranopia':
            colours = ["slategray", "orange", "dodgerblue"]
        elif color_blind == 'tritanopia':
            colours = ["#ff0066", "#ffe6f2", "#00e6e6"]
        cmap1 = colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colours)))
        sm = cm.ScalarMappable(cmap=cmap1, norm=my_norm)
        sm.set_array([])

        if plot_mode == "default":
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('CU', rotation=0, labelpad=25)
            plt.xlabel("CI")
            for m in range(nfeatures):
                ax.barh(y_pos[m], ci[m], color=cmap1(my_norm(cu[m])),
                    edgecolor="#808080", zorder=2)

        if plot_mode == "overlap":
            plt.xlabel("CI and relative CU")
            for m in range(nfeatures):
                ax.barh(y_pos[m], ci[m], color=color_fill_ci,
                        edgecolor=color_edge_ci, linewidth=1.5, zorder=2)
                ax.barh(y_pos[m], cu[m]*ci[m], color=color_fill_cu,
                        edgecolor=color_edge_cu, linewidth=1.5, zorder=2)

        if plot_mode == "combined":
            plt.xlabel("CI and relative CU")
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('CU', rotation=0, labelpad=25)
            for m in range(nfeatures):
                ax.barh(y_pos[m], ci[m], color="#ffffff66", edgecolor="#808080", zorder=2)
                ax.barh(y_pos[m], cu[m]*ci[m], color=cmap1(my_norm(cu[m])), zorder=2)

        plt.ylabel("Features")
        ax.set_xlim(0, CImax)
        if main is not None:
            plt.title(main)

        ax.set_facecolor(color="#D9D9D9")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.grid(which = 'minor')
        ax.grid(which='minor', color='white')
        ax.grid(which='major', color='white')

    def plot_influence(self, ciu_result, xminmax=None, main=None, figsize=(6, 4), colors=("firebrick","steelblue"), 
                       edgecolors=("#808080","#808080")):

        """
        Plot CIU result as a bar plot using Contextual influence values. 

        :param ciu_result: CIU result DataFrame as returned by one of the "explain..." methods. 
        :param xminmax: Range to pass to ``xlim``. Default: None.
        :param figsize: Value to pass as ``figsize`` parameter. Default: (6, 6).
        :param colors: Bar colors to use. Default: ("firebrick","steelblue").
        :param edgecolors: Bar edge colors to use. Default: ("firebrick","steelblue").
        """

        feature_names = ciu_result.feature
        cinfl = ciu_result.Cinfl
        nfeatures = len(feature_names)

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(nfeatures)

        cinfl, feature_names = (list(t) for t in zip(*sorted(zip(cinfl, feature_names))))

        plt.xlabel("ϕ")

        for m in range(len(cinfl)):
            ax.barh(y_pos[m], cinfl[m], color=[colors[0] if cinfl[m] < 0 else colors[1]],
                    edgecolor=[edgecolors[0] if cinfl[m] < 0 else edgecolors[1]], zorder=2)

        plt.ylabel("Features")
        if xminmax is not None:
            ax.set_xlim(xminmax)
        if main is not None:
            plt.title(main)

        ax.set_facecolor(color="#D9D9D9")

        # Y axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.grid(which = 'minor')
        ax.grid(which='minor', color='white')
        ax.grid(which='major', color='white')

    def textual_explanation(self, ciu_result, target_ciu=None, thresholds_ci=None, thresholds_cu=None, use_markdown_effects=False):
        """
        :param dict thresholds_ci: dictionary containing the label and ceiling value for the CI thresholds
        :param dict thresholds_cu: dictionary containing the label and ceiling value for the CU thresholds

        :return: Explanation as `str`.
        """

        if thresholds_ci is None:
            thresholds_ci = {
                'very low importance': 0.20,
                'low importance': 0.40,
                'normal importance': 0.60,
                'high importance': 0.80,
                'very high importance': 1
            }

        if thresholds_cu is None:
            thresholds_cu = {
                'low utility': 0.25,
                'lower than average utility': 0.5,
                'higher than average utility': 0.75,
                'high utility': 1
            }

        if len(thresholds_cu) < 2 or len(thresholds_ci) < 2:
            raise ValueError(f"The dictionaries containing the CI/CU thresholds must have at least 2 elements. \
                             \nCI dict: {thresholds_ci} \nCU dict: {thresholds_cu}")

        # Definitions for text formatting.
        if use_markdown_effects:
            BR = "<br>"
            BLD = "**"
            ITS = "*"
        else:
            BR = "\n"
            BLD = ""
            ITS = ""

        feature_names = ciu_result.loc[:,'feature']
        explanation = []

#            cu_concept = round(self.cu[target_concept] * 100, 2)
        out_name = ciu_result.loc[:,'outname'][0]
        if ciu_result.loc[:,'target_concept'][0] is None:
            outval = ciu_result.loc[:,'outval'][0]
            outmin = self.out_minmaxs.loc[out_name,:][0]
            out_cu = (outval - outmin)/(self.out_minmaxs.loc[out_name,:][1] - outmin)
            out_cu_text = list(thresholds_cu.keys())[self._find_interval(out_cu, thresholds_cu.values())]
            explanation.append(f"The explained value is {BLD}{ITS}{out_name}{ITS}{BLD} with the value " \
                                     f"{outval:.2f} (CU={out_cu:.2f}), which is {BLD}{out_cu_text}{BLD}.{BR}")
        else:
            target_concept = ciu_result.loc[:,'target_concept'][0]
            if target_ciu is not None:
                ci = target_ciu.loc[target_concept,'CI'][0]
                ci_text = list(thresholds_ci.keys())[self._find_interval(ci, thresholds_ci.values())]
                cu = target_ciu.loc[target_concept,'CU'][0]
                cu_text = list(thresholds_cu.keys())[self._find_interval(cu, thresholds_cu.values())]
                explanation.append(f"The explained value is {BLD}{ITS}{target_concept}{ITS}{BLD} for output "\
                                   f"{BLD}{ITS}{out_name}{ITS}{BLD}, which has "\
                                   f"{BLD}{ci_text} (CI={ci:.2f}){BLD} and {BLD}{cu_text} (CU={cu:.2f}){BLD}.{BR}")
            else: 
                explanation.append(f"The explained value is {BLD}{ITS}{target_concept}{ITS}{BLD} for output {BLD}{ITS}{out_name}{ITS}{BLD}.{BR}")

        for feature in list(feature_names):
            ci = ciu_result.loc[feature,'CI'][0]
            ci_text = list(thresholds_ci.keys())[self._find_interval(ci, thresholds_ci.values())]
            cu = ciu_result.loc[feature,'CU'][0]
            cu_text = list(thresholds_cu.keys())[self._find_interval(cu, thresholds_cu.values())]
            fvalue = ciu_result.loc[feature,'invals'][0]
            if len(fvalue) == 1: # Coalition or single feature?
                fvalue = fvalue[0]
            explanation.append(f"Feature {ITS}{feature}{ITS} has {BLD}{ci_text} (CI={ci:.2f}){BLD} " \
                               f"and has value(s) {fvalue}, which is {BLD}{cu_text} (CU={cu:.2f}){BLD}{BR}")

        return "".join(explanation)

    def _find_interval(self, value, thresholds):
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                return i
        return len(thresholds) - 1 # We can't allow indices to go beyond this. 

    def plot_3D(self, ind_inputs, instance=None, ind_output=0, nbr_pts = (40,40), figsize=(6,6), 
                azim=None):
        """
        Plot output value as a function of two inputs. 

        :param ind_inputs: indexes for two features to use for the 3D plot.
        :type ind_inputs: list
        :param instance: instance to use if it is different from given to the constructor or 
            as a parameter to ``explain()``or `èxplain_voc()``. Default: None.
        :type instance: pd.DataFrame
        :param ind_output: index of output to plot. Default: 0.
        :type ind_output: int
        :param nbr_pts: number of points to use (both axis). Default: (40,40).
        :type nbr_pts: int
        :param figsize: Values to pass to ``plt.figure()``. Default: (6,6).
        :param azim: azimuth angle to use. Default: None.
        :return: 3D plot object
        """
        # Deal with None parameters and other parameter value arrangements.
        if instance is None:
            instance = self.instance
 
        # Get input/feature names
        fnames = [self.input_names[i] for i in ind_inputs]

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Generate data points
        minmaxs = self.in_minmaxs
        x = np.linspace(minmaxs.iloc[ind_inputs[0],0], minmaxs.iloc[ind_inputs[0],1], nbr_pts[0])
        y = np.linspace(minmaxs.iloc[ind_inputs[1],0], minmaxs.iloc[ind_inputs[1],1], nbr_pts[1])
        x, y = np.meshgrid(x, y)
        total_npoints = x.shape[0]*x.shape[1]
        m = np.tile(instance, (total_npoints, 1))
        m[:,ind_inputs[0]] = x.reshape(total_npoints)
        m[:,ind_inputs[1]] = y.reshape(total_npoints)
        z = self.predictor(pd.DataFrame(m, columns=self.input_names))
        zm = z[:,ind_output].reshape(x.shape[0], x.shape[1])

        # Create a 3D surface plot
        ax.plot_surface(x, y, zm, color="lightblue", linewidth=1, antialiased=True, zorder=1, alpha=0.8)

        # Adding instance point marker
        outvals = self.predictor(instance)
        if outvals.ndim == 1:
            outvals = outvals[:,np.newaxis]
        xp = instance.iloc[0, ind_inputs[0]]
        yp = instance.iloc[0, ind_inputs[1]]
        ax.scatter(xp, yp, outvals[0,ind_output], color="red", alpha=1, s=100, zorder=3)
        fig.suptitle(f"Prediction for {self.out_names[ind_output]} is {outvals[0,ind_output]}")

        # Add labels
        ax.set_xlabel(fnames[0])
        ax.set_ylabel(fnames[1])
        ax.set_zlabel(self.out_names[ind_output])

        # Final adjustments
        if azim is not None:
            ax.azim = azim

