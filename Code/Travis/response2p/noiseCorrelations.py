import numpy as np
import pandas as pd

def noiseCorrelations(unmixed, isResponsive, useAll = False, numBaseline=15, window = [15,45]):
    if useAll:    
        neurons = list(isResponsive[isResponsive['isResponsive']==True]['neuron'])
    else:
        neurons = list(isResponsive[isResponsive['isResponsive']==True]['neuron'])
    
    unmixed = unmixed[neurons,window[0]:window[1],:]
    unmixed_mean = unmixed.mean(axis=-1)
    bl_subt = unmixed - np.expand_dims(unmixed_mean,axis=-1)
    corrmaps = []

    numNeurons, numTimepoints, numFreqs, numAttens, numTrials = unmixed.shape

    for freq in range(unmixed.shape[2]):
        for atten in range(unmixed.shape[3]):
            #reshape so that each trial is in it's own row
            temp = np.reshape(bl_subt[:,:,freq,atten,:],(numNeurons,numTrials*numTimepoints), order='F')
            temp = np.reshape(temp, (numNeurons*numTrials, numTimepoints))
            corrs = np.corrcoef(temp)
            corrmap = np.zeros((numNeurons, numNeurons, numTrials))
            for i, n1 in enumerate(neurons):
                for j, n2 in enumerate(neurons):
                    if j != i and j > i:
                        start_i = numTrials*i
                        start_j = numTrials*j
                        corrmap[i,j,:] = np.diagonal(corrs[start_i:start_i+10, start_j:start_j+10])
            corrmaps.append(corrmap)

    corrmaps = np.stack(corrmaps, axis=-1) ##this is a neuron x neuron x trial x freq/atten matrix of correlations, upper triangle 
    corrmap_avg = corrmaps.mean(axis=(-1,-2))
    
    neuron1 = []
    neuron2 = []
    corrs = []
    for i, n1 in enumerate(neurons):
        for j, n2 in enumerate(neurons):
            if i != j & j > i:
                neuron1.append(n1)
                neuron2.append(n2)
                corrs.append(corrmap_avg[i,j])

    df = pd.DataFrame({'neuron1':neuron1, 'neuron2':neuron2, 'noise_corrs':corrs})
    return df