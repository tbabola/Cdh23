import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")),"Code","Travis"))
from response2p import signalCorrelations, noiseCorrelations
import responses_analysis

def getData(twop_mice, verbose = False, force = False):
    sig_responding_neurons = []
    total_neurons = []
    meancorrs = []
    noisecorrs = []

    for index, row in twop_mice.iterrows():
        if verbose: print(row['Mouse'])

        analysis_dir = row['2P_path'] / "response_analysis"
        if not analysis_dir.exists():
            (row['2P_path'] / "response_analysis").mkdir()

        if (analysis_dir / "unmixed.npy").exists():
            unmixed = np.load(analysis_dir / "unmixed.npy")
        else:
            temp = dfof.processFluor(dir = row['2P_path'])
            temp.loadTraces()
            temp.calc_dfof(window=500, step = 50, percentile=10, cutoff = 20)
            temp.unmix(numBaseline=15, framesAfter=60)
            np.save(analysis_dir / "unmixed.npy", temp.unmixed)

        responsive_file = analysis_dir / "isResponsive.csv"
        if responsive_file.exists() and not force:
            isResponsive = pd.read_csv(responsive_file)
        else:
            isResponsive = responses_analysis.isResponsive(unmixed, alpha = 0.01)
            isResponsive.to_csv(responsive_file, index=False)

        ####now which responses are sound responsive?
        ## signalCorrelations
        signalcorr_file = analysis_dir / "signalCorrelations.csv" 
        if signalcorr_file.exists() and not force:
            signalcorr = pd.read_csv(signalcorr_file)
        else:
            if verbose: print("running signal correlations")
            signalcorr = signalCorrelations.signalCorrelations(unmixed, isResponsive)
            signalcorr.to_csv(signalcorr_file, index=False)

        ##noise correlations
        noisecorr_file = analysis_dir / "noiseCorrelations.csv" 
        if noisecorr_file.exists() and not force:
            noisecorr = pd.read_csv(noisecorr_file)
        else:
            if verbose: print("running noise correlations")
            noisecorr = noiseCorrelations.noiseCorrelations(unmixed, isResponsive)
            noisecorr.to_csv(noisecorr_file, index=False)

        
        sig_responding_neurons.append(isResponsive['isResponsive'].value_counts().loc[True])
        total_neurons.append(isResponsive.shape[0])
        meancorrs.append(signalcorr['corrs'].mean())
        noisecorrs.append(noisecorr['noise_corrs'].mean())

    twop_mice['total_neurons'] = total_neurons
    twop_mice['sig_responders'] = sig_responding_neurons
    twop_mice['sound_responsive_percent'] = twop_mice['sig_responders'] / twop_mice['total_neurons'] * 100
    twop_mice['signalCorr'] = meancorrs
    twop_mice['noiseCorr'] = noisecorrs

    return twop_mice