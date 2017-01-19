import numpy as np

def apk12(batch, prediction):
    if len(batch) == 5:
        _, y, g, r, _ = batch
    else:
	_, y, g, r, _, _ = batch
    yp = []
    result = []
    if True:
	for i in range(len(r)-1):
            start, end = r[i], r[i+1]
    	    yp.append(prediction[i,0:end-start])
	    act = [c for c,i in enumerate(y[start:end]) if i>0]
            pred = {i:c for c,i in enumerate(yp[-1])}
	    pred = [pred[i] for i in sorted(yp[-1],reverse=True)]
	    result.append(apk(act,pred,k=12))
	yp = np.concatenate(yp)
    assert(len(y)==len(yp))
    return np.mean(result), yp
    


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]
    if not actual:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)


    return score / min(len(actual), k)

