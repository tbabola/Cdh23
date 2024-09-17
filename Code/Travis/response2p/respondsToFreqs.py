import pandas as pd

def respondsToFreqs(df, rejectAtten = None):
    neurons = []
    respondingFreqsBinary = []

    if rejectAtten:
        df = df[~df['atten'].isin(rejectAtten)]

    for neuron in df['neuron'].unique():
        temp = df[df['neuron']==neuron]
        if temp['reject_hs'].any():
            respondingFreqs = temp[temp['reject_hs']]['freq'].unique()
            tempBinary = []
            neurons.append(neuron)
            for i in range(len(temp['freq'].unique())):
                if i in respondingFreqs:
                    tempBinary.append(1)
                else:
                    tempBinary.append(0)
            respondingFreqsBinary.append(tempBinary)
            
    freqResponders = pd.DataFrame({'respondingFreqs':respondingFreqsBinary})
    columns = ["freq" + str(freq) + "_response" for freq in df['freq'].unique()]
    freqResponders = pd.DataFrame(freqResponders['respondingFreqs'].to_list(), columns=columns)
    freqResponders['neuron'] = neurons
    #freqResponders['neuron'] = pd.to_numeric(freqResponders['neuron'])

    return freqResponders