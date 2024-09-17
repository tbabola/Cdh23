import numpy as np
import pandas as pd

def signalCorrelations(unmixed, isResponsive, numBaseline=15, window = [15,45]):
    responsiveNeurons = list(isResponsive[isResponsive['isResponsive']==True]['neuron'])
    unmixed_mean = unmixed.mean(axis=-1)
    bl_subt = unmixed_mean[:,window[0]:window[1],:] - unmixed_mean[:,:numBaseline,:].mean(axis=1, keepdims=True)

    corrmaps = []
    for freq in range(unmixed.shape[2]):
        for atten in range(unmixed.shape[3]):
            corrmaps.append(np.corrcoef(bl_subt[responsiveNeurons,:,freq,atten]))
    corrmap = np.stack(corrmaps).mean(axis=0)   

    neuron1 = []
    neuron2 = []
    corrs = []
    for i, n1 in enumerate(responsiveNeurons):
        for j, n2 in enumerate(responsiveNeurons):
            if i != j & j < i:
                neuron1.append(n2)
                neuron2.append(n1)
                corrs.append(corrmap[i,j])

    df = pd.DataFrame({'neuron1':neuron1, 'neuron2':neuron2, 'corrs':corrs})
    return df