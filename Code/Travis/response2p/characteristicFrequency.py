import pandas as pd
def characteristicFrequency(df, rejectAtten = None, rejectFreq = None):
    neurons = []
    cf = []
    cf_atten = []
    cf_amp = []

    if rejectAtten:
        df = df[~df['atten'].isin(rejectAtten)]
    if rejectFreq:
        df = df[~df['freq'].isin(rejectFreq)]

    for neuron in df['neuron'].unique():
        temp = df[df['neuron']==neuron]
        if temp['reject_hs'].any():
            maxAtten = temp[temp['reject_hs']]['atten'].max()
            temp2 = temp[(temp['atten']==maxAtten) & (temp['reject_hs']==True)]['response_amp'].idxmax()
            neurons.append(neuron)
            cf.append(temp.loc[temp2,'freq'])
            cf_atten.append(temp.loc[temp2,'atten'])
            cf_amp.append(temp.loc[temp2,'response_amp'])


    cf_df = pd.DataFrame({'neuron':neurons, 'charFreq': cf, 'charFreq_atten':cf_atten, 'charFreq_amp':cf_amp})
    return cf_df