import numpy as np


def equalized_odds_measure_TP(data, model, sensitive_features, ylabel, rev_pred=1):
    '''
    True positive label for the groups defined by the values of the "sensible_features",
    with respect to the "model" on the "data".
    :param data: the data where to evaluate the True Positive Rate (Equal Opportunity).
    :param model:  the model that has to be evaluated.
    :param sensitive_features: the features used to split the data in groups.
    :param ylabel: the POSITIVE label (usually +1).
    :param rev_pred: an option to reverse the outputs of our model.
    :return: a dictionary with keys the sensitive_features and values dictionaries containing the True Positive Rates
    of the different subgroups of the specific sensitive feature.
    '''
    predictions = model.predict(data.data) * rev_pred
    print("average predict ones:",np.mean(predictions))
    truth = data.target
    eq_dict = {}
    for val in [0,1]:
        positive_sensitive = np.sum([1.0 if sensitive_features[i] == val and truth[i] == ylabel else 0.0
                                         for i in range(len(predictions))])
        eq_dict[val] =None
        if positive_sensitive > 0:
            eq_tmp = np.sum([1.0 if predictions[i] == ylabel and sensitive_features[i] == val and truth[i] == ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive
        eq_dict[val] = eq_tmp
   
    return eq_dict[1]-eq_dict[0]
