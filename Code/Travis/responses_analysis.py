import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy
from pathlib import Path

def get2Pmice(data_path, folderList):
    ## returns a dataframe with mouse list merged with appropriate 2P path for L23 imaging
    mice = []
    folders = []
    for folder in folderList:
        mice.append(Path(folder).parents[2].name)
        folders.append(Path(folder))
    twop_paths = pd.DataFrame({"Mouse":mice, "2P_path":folders})

    mouseInfo = pd.read_csv(data_path / "MouseInfo.csv")

    return pd.merge(mouseInfo, twop_paths, on='Mouse')

def isResponsive(data, numBaseline = 15, response = [23,27], alpha = 0.05):
    baseline = data[:,0:15,:].mean(axis=1)
    response = data[:,23:27,:].mean(axis=1)

    neuron_index = []
    sigs = []

    for i in range(baseline.shape[0]):
        freqs = []
        attens = []
        response_var = []
        response_amp = []
        for freq in range(baseline.shape[1]):
            for atten in range(baseline.shape[2]):
                for repeat in range(baseline.shape[3]):
                    freqs.extend([freq]*2)
                    attens.extend([atten]*2)
                    response_var.extend([0,1])
                    response_amp.append(baseline[i,freq,atten,repeat])
                    response_amp.append(response[i,freq,atten,repeat])

        df = pd.DataFrame({"freq":freqs,"atten":attens, "response_var":response_var, "response_amp":response_amp })
        # Performing two-way ANOVA
        model = ols(
            'response_amp ~ C(response_var) + C(freq) + C(atten) + C(freq):C(atten):C(response_var)', data=df).fit()
        #print(sm.stats.anova_lm(model, typ=2)['PR(>F)'][0])


        sigs.append(sm.stats.anova_lm(model, typ=2)['PR(>F)']['C(response_var)'])
        neuron_index.append(i)

    isResponsive_df = pd.DataFrame({'neuron':neuron_index,"PR(>F)":sigs})
    isResponsive_df['isResponsive'] = isResponsive_df['PR(>F)'] < alpha
    return isResponsive_df

def isResponsive_withoffset(data, numBaseline = 15, onset_response = [20,24], offset_response = [26,30], alpha = 0.05):
    baseline = data[:,:numBaseline,:].mean(axis=1)
    onset = data[:,onset_response[0]:onset_response[1],:].mean(axis=1)
    offset = data[:,offset_response[0]:offset_response[1],:].mean(axis=1)

    neuron_index = []
    sigs = []

    for i in range(baseline.shape[0]):
        freqs = []
        attens = []
        response_var = []
        response_amp = []
        for freq in range(baseline.shape[1]):
            for atten in range(baseline.shape[2]):
                for repeat in range(baseline.shape[3]):
                    freqs.extend([freq]*3)
                    attens.extend([atten]*3)
                    response_var.extend([0,1,2])
                    response_amp.append(baseline[i,freq,atten,repeat])
                    response_amp.append(onset[i,freq,atten,repeat])
                    response_amp.append(offset[i,freq,atten,repeat])

        df = pd.DataFrame({"freq":freqs,"atten":attens, "response_var":response_var, "response_amp":response_amp })
        # Performing two-way ANOVA
        model = ols(
            'response_amp ~ C(response_var) + C(freq) + C(atten) + C(freq):C(atten):C(response_var)', data=df).fit()
        #print(sm.stats.anova_lm(model, typ=2)['PR(>F)'][0])

        sigs.append(sm.stats.anova_lm(model, typ=2)['PR(>F)']['C(response_var)'])
        neuron_index.append(i)

    isResponsive_df = pd.DataFrame({'neuron':neuron_index,"PR(>F)":sigs})
    isResponsive_df['isResponsive'] = isResponsive_df['PR(>F)'] < alpha
    return isResponsive_df

def getResponses(data, windows = None):
    """
    Returns a dataframe with responses defined by windows as variables on a frequency,
    attenuation, and trial basis. Useful for running statistical tests

    Parameters
    ----------
    data : a neuron x time x frequency x attenuation x trial array

    windows : a list of tuples that define the time windows to examine, will be averaged to give final value 
    """

    try:
        numNeurons, timepoints, numFreqs, numAttens, numTrials = data.shape
    except:
        print("The data array is the incorrect shape. Format to neuron x time x frequency x attenuation x trial")
    try:
        numWindows = len(windows)
    except:
        print("There are no windows defined.")

    neurons = []
    freqs = []
    attens = []
    response_var = []
    response_amp = []
    responses = []

    for window in windows:
        responses.append(data[:,window[0]:window[1],:].mean(axis=1))
    responses = np.stack(responses,axis=-1)

    for neuron in range(numNeurons):
        for freq in range(numFreqs):
            for atten in range(numAttens):
                for repeat in range(numTrials):
                    for window in range(numWindows):
                        neurons.append(neuron)
                        freqs.append(freq)
                        attens.append(atten)
                        response_var.append(window)
                        response_amp.append(responses[neuron,freq,atten,repeat, window])

    df = pd.DataFrame({"neuron": neurons, "freq":freqs,"atten":attens, "response_var":response_var, "response_amp":response_amp })
    return df    

def onset_offset_classification(responses):
    neurons = []
    onset_pvalue = []
    onset_amps = []
    offset_pvalue = []
    offset_amps = []
    onset_offset_pvalue = []

    for neuron in responses['neuron'].unique():
        temp = responses[responses['neuron']==neuron]
        baseline = temp[temp['response_var']==0]
        baseline_amp = baseline['response_amp'].mean()
        onset = temp[temp['response_var']==1]
        onset_amp = onset['response_amp'].mean()
        offset = temp[temp['response_var']==2]
        offset_amp = offset['response_amp'].mean()

        neurons.append(neuron)
        ## t-tests, in all cases we want to test if response is higher than the baseline, don't bother with negatively deflecting cells
        onset_pvalue.append(scipy.stats.ttest_rel(baseline['response_amp'], onset['response_amp'],alternative='less').pvalue)
        onset_amps.append(onset_amp-baseline_amp)
        offset_pvalue.append(scipy.stats.ttest_rel(baseline['response_amp'], offset['response_amp'], alternative='less').pvalue)
        offset_amps.append(offset_amp-baseline_amp)
        onset_offset_pvalue.append((scipy.stats.ttest_rel(onset['response_amp'], offset['response_amp'], alternative='less').pvalue))

    df = pd.DataFrame({"neuron":neurons, "onset_amp":onset_amps, "onset_pvalue":onset_pvalue, "offset_amp":offset_amps, 'offset_pvalue': offset_pvalue, "onset_offset_pvalue": onset_offset_pvalue})

    ###classification of responses
    df['onset_sig'] = df['onset_pvalue'] < (0.05/3)  # bonferroni correction since 3 t-tests
    df['offset_sig'] = df['offset_pvalue'] < (0.05/3)
    df['offset_gt_onset_sig'] = df['onset_offset_pvalue'] < (0.05/3)
    df['Classification'] = None

    onset_index = (df['onset_sig']) & ~(df['offset_sig'])
    df.loc[onset_index,'Classification'] = "Onset"
    offset_index = ~(df['onset_sig']) & (df['offset_sig'])
    df.loc[offset_index,'Classification'] = "Offset"
    onset_offset_index = (df['onset_sig']) & (df['offset_sig']) & (df['offset_gt_onset_sig'])
    df.loc[onset_offset_index,'Classification'] = "Onset/Offset"
    onset2_index = (df['onset_sig']) & (df['offset_sig']) & ~(df['offset_gt_onset_sig'])
    df.loc[onset2_index,'Classification'] = "Onset"

    return df







        