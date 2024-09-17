from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import pandas as pd


def getResponses(data, numBaseline = 15, response = [23,27]):
    baseline = data[:,0:numBaseline,:].mean(axis=1)
    response = data[:,response[0]:response[1],:].mean(axis=1)
    neuron_index = []
    freqs = []
    attens = []
    response_var = []
    response_amp = []

    for neuron in range(baseline.shape[0]):
        for freq in range(baseline.shape[1]):
            for atten in range(baseline.shape[2]):
                for repeat in range(baseline.shape[3]):
                    neuron_index.extend([neuron]*2)
                    freqs.extend([freq]*2)
                    attens.extend([atten]*2)
                    response_var.extend([0,1])
                    response_amp.append(baseline[neuron,freq,atten,repeat])
                    response_amp.append(response[neuron,freq,atten,repeat])

    print(len(neuron_index), len(freqs), len(attens), [str(neuron)]*2)
    df = pd.DataFrame({"neuron": neuron_index, "freq":freqs,"atten":attens, "response_var":response_var, "response_amp":response_amp })
    return df

def responseStats(df, isResponsive):
    df_stats = pd.DataFrame()
    for index, row in isResponsive.iterrows():
        if row['isResponsive']:
            neurons = []
            freqs = []
            attens = []
            pvals = []
            mean_response = []

            for freq in df['freq'].unique():
                for atten in df['atten'].unique():
                    temp_df = df[(df['neuron']==row['neuron'])& (df['freq']==freq) & (df['atten']==atten)]
                    baseline = temp_df[temp_df['response_var']==0]
                    response = temp_df[temp_df['response_var']==1]
                    neurons.append(row['neuron'])
                    freqs.append(freq)
                    attens.append(atten)
                    pvals.append(ttest_rel(baseline['response_amp'],response['response_amp'],alternative='less').pvalue)
                    mean_response.append(response['response_amp'].mean(axis=0) - baseline['response_amp'].mean(axis=0))

            df_response = pd.DataFrame({'neuron':neurons, 'freq':freqs, 'atten': attens, 'pval':pvals, 'response_amp':mean_response})
            mt = multipletests(df_response['pval'], alpha = 0.20, method="fdr_bh")
            df_response['reject_hs'] = mt[0]
            df_response['pval_corrected'] = mt[1]

            df_stats = pd.concat((df_stats, df_response), ignore_index=True)
    return df_stats