import pandas as pd

def bestFrequency(df, rejectAtten = None):
    neurons = []
    bf = []
    bf_atten = []

    if rejectAtten:
        df = df[~df['atten'].isin(rejectAtten)]

    for neuron in df['neuron'].unique():
        temp = df[df['neuron']==neuron]
        if temp['reject_hs'].any():
            bf_index = temp[temp['reject_hs']==True]['response_amp'].idxmax()

            neurons.append(neuron)
            bf.append(temp.loc[bf_index,'freq'])
            bf_atten.append(temp.loc[bf_index,'atten'])

    bf_df = pd.DataFrame({'neuron':neurons, 'bestFreq': bf, 'bestFreq_atten':bf_atten})
    return bf_df 