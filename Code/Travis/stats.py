import itertools
import scipy
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np



def ttests(data, iv_cat, dv_cat, type = 'mean'):
    #iv_2: independent variable, frequency
    #dv_cat: dependent variable, proportions
    genotypes = data['Genotype'].unique()
    combos = itertools.combinations(genotypes,2)
    ivs = data[iv_cat].unique()

    pvalues = []
    geno1s = [] 
    geno2s = []
    iv_list = []
    mean1 = []
    std1 = []
    mean2 = []
    std2 = []

    for combo in combos:
        for iv in ivs:
            temp1 = data[(data['Genotype']==combo[0]) & (data[iv_cat]==iv)]
            temp2 = data[(data['Genotype']==combo[1]) & (data[iv_cat]==iv)]
            pvalue = scipy.stats.ttest_ind(temp1[dv_cat], temp2[dv_cat]).pvalue
            if np.isnan(pvalue):
                pvalue = 1
            pvalues.append(pvalue)
            geno1s.append(combo[0])
            geno2s.append(combo[1])
            iv_list.append(iv)
            if type == 'mean':
                if temp1[dv_cat].mean():
                    mean1.append(temp1[dv_cat].mean())
                    std1.append(temp1[dv_cat].std())
                else:
                    mean1.append(0)
                    std1.append(0)
                if temp2[dv_cat].mean():
                    mean2.append(temp2[dv_cat].mean())
                    std2.append(temp2[dv_cat].std())
                else:
                    mean2.append(0)
                    std2.append(0)
                dev_str = 'std'
                
            elif type == 'median':
                mean1.append(temp1[dv_cat].median())
                mean2.append(temp2[dv_cat].median())
                std1.append(scipy.stats.iqr(temp1[dv_cat]))
                std2.append(scipy.stats.iqr(temp2[dv_cat]))
                dev_str = "IQR"



    stats_df = pd.DataFrame({"geno1":geno1s, "geno2": geno2s, iv_cat:iv_list, str(type + "1"):mean1, str(dev_str + "1"): std1, str(type + "2"):mean2, str(dev_str + "2"): std2, "pval":pvalues})
    mt = multipletests(stats_df['pval'], alpha = 0.05, method="fdr_bh")
    stats_df['reject_hs'] = mt[0]
    stats_df['pval_corrected'] = mt[1]
    return stats_df

def ttests_2cats(data, iv_cat, iv_cat2, dv_cat):
    #iv_2: independent variable, frequency
    #dv_cat: dependent variable, proportions
    genotypes = data['Genotype'].unique()
    combos = itertools.combinations(genotypes,2)
    ivs = data[iv_cat].unique()
    ivs2 = data[iv_cat2].unique()

    pvalues = []
    geno1s = [] 
    geno2s = []
    iv_list = []
    iv2_list = []
    mean1, std1, mean2, std2 = ([],[],[],[])

    for combo in combos:
        print(combo)
        print(ivs)
        for iv in ivs:
            for iv2 in ivs2:
                temp1 = data[(data['Genotype']==combo[0]) & (data[iv_cat]==iv) & (data[iv_cat2]==iv2)]
                temp2 = data[(data['Genotype']==combo[1]) & (data[iv_cat]==iv) & (data[iv_cat2]==iv2)]
                pvalues.append(scipy.stats.ttest_ind(temp1[dv_cat], temp2[dv_cat]).pvalue)
                geno1s.append(combo[0])
                geno2s.append(combo[1])
                iv_list.append(iv)
                iv2_list.append(iv2)
                mean1.append(temp1[dv_cat].mean())
                mean2.append(temp2[dv_cat].mean())
                std1.append(temp1[dv_cat].std())
                std2.append(temp2[dv_cat].std())

    stats_df = pd.DataFrame({"geno1":geno1s, "geno2": geno2s, iv_cat:iv_list, iv_cat2:iv2_list, "mean1":mean1, "std1": std1, "mean2":mean2, "std2": std2, "pval":pvalues})
    mt = multipletests(stats_df['pval'], alpha = 0.05, method="fdr_bh")
    stats_df['reject_hs'] = mt[0]
    stats_df['pval_corrected'] = mt[1]
    return stats_df

def ttests_2(data, dv_cat):
    genotypes = data['Genotype'].unique()
    combos = itertools.combinations(genotypes,2)

    pvalues = []
    geno1s = [] 
    geno2s = []

    for combo in combos:
        temp1 = data[(data['Genotype']==combo[0])]
        temp2 = data[(data['Genotype']==combo[1])]
        pvalues.append(scipy.stats.ttest_ind(temp1[dv_cat], temp2[dv_cat]).pvalue)
        geno1s.append(combo[0])
        geno2s.append(combo[1])


    stats_df = pd.DataFrame({"geno1":geno1s, "geno2": geno2s, "pval":pvalues})
    mt = multipletests(stats_df['pval'], alpha = 0.05, method="fdr_bh")
    stats_df['reject_hs'] = mt[0]
    stats_df['pval_corrected'] = mt[1]
    return stats_df

def mannwhitneys(data, dv_cat):
    genotypes = data['Genotype'].unique()
    combos = itertools.combinations(genotypes,2)

    pvalues = []
    geno1s = [] 
    geno2s = []

    for combo in combos:
        temp1 = data[(data['Genotype']==combo[0])]
        temp2 = data[(data['Genotype']==combo[1])]
        pvalues.append(scipy.stats.mannwhitneyu(temp1[dv_cat], temp2[dv_cat]).pvalue)
        geno1s.append(combo[0])
        geno2s.append(combo[1])


    stats_df = pd.DataFrame({"geno1":geno1s, "geno2": geno2s, "pval":pvalues})
    mt = multipletests(stats_df['pval'], alpha = 0.05, method="fdr_bh")
    stats_df['reject_hs'] = mt[0]
    stats_df['pval_corrected'] = mt[1]
    return stats_df

def ttests_within(data, dv_cat, estimator = 'mean'):
    #iv_2: independent variable, frequency
    #dv_cat: dependent variable, proportions
    genotypes = data['Genotype'].unique()
    combos = itertools.combinations(genotypes,2)

    pvalues = []
    geno1s = [] 
    geno2s = []
    mean1 = []
    std1 = []
    mean2 = []
    std2 = []

    for combo in combos:
        print(combo)
        temp1 = data[(data['Genotype']==combo[0])]
        temp2 = data[(data['Genotype']==combo[1])]
        pvalue = scipy.stats.ttest_ind(temp1[dv_cat], temp2[dv_cat]).pvalue
        if np.isnan(pvalue):
            pvalue = 1
        pvalues.append(pvalue)
        geno1s.append(combo[0])
        geno2s.append(combo[1])
        if estimator == 'mean':
            mean1.append(temp1[dv_cat].mean())
            mean2.append(temp2[dv_cat].mean())
            std1.append(temp1[dv_cat].std())
            std2.append(temp2[dv_cat].std())
            dev_str = 'std'
        elif estimator == 'median':
            mean1.append(temp1[dv_cat].median())
            mean2.append(temp2[dv_cat].median())
            std1.append(scipy.stats.iqr(temp1[dv_cat]))
            std2.append(scipy.stats.iqr(temp2[dv_cat]))
            dev_str = "IQR"
        else:
            print("NO ESTIMATOR")

    stats_df = pd.DataFrame({"geno1":geno1s, "geno2": geno2s, str(estimator + "1"):mean1, str(dev_str + "1"): std1, str(estimator + "2"):mean2, str(dev_str + "2"): std2, "pval":pvalues})
    mt = multipletests(stats_df['pval'], alpha = 0.05, method="fdr_bh")
    stats_df['reject_hs'] = mt[0]
    stats_df['pval_corrected'] = mt[1]
    return stats_df
