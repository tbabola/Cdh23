import pandas as pd
import numpy as np


def bandwidth(data, attenLevel = 1, mustBeContinuous = False, freqDict = {0:4000, 1:8000, 2:16000, 3:32000, 4:64000}):
    df = data[(data['atten']==attenLevel) & (data['reject_hs']==True)].copy()
    neurons = []
    bandwidths = []

    for neuron in df['neuron'].unique():
        temp = df[df['neuron']==neuron]
        min_freq = temp['freq'].min()
        max_freq = temp['freq'].max()
        bw = np.log2(freqDict[max_freq]/freqDict[min_freq])    

        if mustBeContinuous:
            if (max_freq - min_freq + 1) == len(temp):
                neurons.append(neuron)
                bandwidths.append(bw)
        else:    
            neurons.append(neuron)
            bandwidths.append(bw)
    
    return pd.DataFrame({"neuron":neurons, "bw":bandwidths})