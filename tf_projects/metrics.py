import pandas as pd
import numpy as np


def PPTS(
        records,
        predictions,
        L,
        U=100
):
    records = np.array(records)
    # arrange the records into descending order.
    measurements = records[np.argsort(records)[::-1]]
    # arrange the predictions due to descending index of records.
    predictions = predictions[np.argsort(records)[::-1]]

    K_L = (L * measurements.size / 100)
    K_U = (U * measurements.size / 100)
    segma = np.abs((predictions - measurements) / measurements)
    segma = segma[int(K_L):int(K_U),:]
    ppts = np.sum(segma)/(K_L-K_U+1)

    return ppts

def NSEC(observations,predictions):
    """ 
    Compute Nash Sutcliffe Coefficient.
    Args:
        observations: the actual observations. Records measurements
        predictions: the predict value.
    """

    observations_mean = np.mean(observations)
    return 1-np.sum(np.power(observations-predictions,2))/np.sum(np.power(observations-observations_mean,2))

np.random.seed(123)
records=np.random.rand(540,1)
noise = np.random.randint(4,8,(540,1))
noise1=np.random.randint(4,6,(540,1))
predictions=records+noise-noise1

print(PPTS(records=records,predictions=predictions,L=95))