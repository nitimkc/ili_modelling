#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2025

@author: nitimkc
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import scipy.stats as stats
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from statsmodels.tsa.stattools import grangercausalitytests

# import statsmodels.api as sm

def corr_sig(df, targets, sig_level=0.05, method='pearson'):
    
    df.dropna(inplace=True)
    t1, t2 = targets
    if method=='pearson':
        corr, p_val =  stats.pearsonr(df[t1], df[t2])
    elif method=='spearman':
        corr, p_val =  stats.spearmanr(df[t1], df[t2])
    else:
        print('provided method not supported')
        
    sig_test = bool(p_val < sig_level)
    return corr, p_val, sig_test

def granger_test(df, targets, max_lag=2, sig_level=0.05):

    g_results = []
    g_test = grangercausalitytests(df[targets], maxlag=max_lag, verbose=0)
    for lag, test in g_test.items():
        all_test = test[0]
        f_test = all_test['ssr_ftest']
        f_stat = f_test[0]
        p_val = f_test[1]
        sig_test = bool(p_val < sig_level)
        g_results.append({f"lag{lag}":(f_stat, p_val, sig_test)})
    return g_results

def cross_correlation_analysis(df, targets, time, country, savepath, max_lag=5, plot=False):
    """
    Compute cross-correlation between two time series over a range of lags.
    Args:
        df: dataframe with two time series
        targets: list of columns names of two time series
        max_lag: Maximum lag to consider in both directions
        plot: Whether to show a plot

    Returns:
        A DataFrame with lag, correlation, and the lag of maximum correlation.
    """
    t1, t2 = targets
    df.dropna(inplace=True)

    lags = range(-max_lag, max_lag + 1)
    corr = [df[t1].corr(df[t2].shift(lag)) for lag in lags]

    result_df = pd.DataFrame({'lag': lags, 'correlation': corr})
    best_lag = result_df.iloc[result_df['correlation'].idxmax()]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.stem(result_df['lag'], result_df['correlation'], use_line_collection=True)
        plt.axvline(x=0, color='gray', linestyle='--', label='Zero Lag')
        plt.axvline(x=best_lag['lag'], color='red', linestyle='--', label=f'Best Lag: {int(best_lag["lag"])}')
        plt.title('Cross-Correlation between Time Series')
        plt.xlabel('Lag (weeks)')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(savepath.joinpath(f"cross_corr_{time}_{country}"))

    return result_df, best_lag

def descriptions(df, targets, country, time, savepath, sig_level=0.05, max_lag=2):
    
    description = {}
    description['country'] = country
    description['time'] = time
    description['pearson_corr'] = corr_sig(df, targets, sig_level, method='pearson')
    description['spearman_corr'] = corr_sig(df, targets, sig_level, method='spearman')
    
    # total cost of aligning one time series with another by allowing for stretching, 
    # compressing, or warping of the time axis — while minimizing point-wise differences
    # 0 = perfect match
    t1, t2 = targets
    dtw_distance, _ = fastdtw(df[t1], df[t2], dist=euclidean)
    description['dtw_distance'] = dtw_distance

    # granger causality test
    description['granger_offtwt'] = granger_test(df, [t1,t2])
    description['granger_twtoff'] = granger_test(df, [t2,t1])

    # cross_correlation
    cross_corr, best_lag = cross_correlation_analysis(df, targets, time, country, savepath)
    best_lag = best_lag[0]
    description['best_lag'] = best_lag
    if best_lag > 0:
            print(f"Official leads Twitter by {abs(best_lag)} for {time} period")
    elif best_lag < 0:
        print(f"Twitter leads Official by {abs(best_lag)} for {time} period")
    else:
        print(f"one does not lead the other: {best_lag} for {time} period")

    # print(description)
    with open(savepath.joinpath(f"descriptions.json"), "a") as f:
        f.write(json.dumps(description) + "\n")
    

def plot_ili(plot_df, targets, savepath, country):
    """
    Plot Real Data vs. Twitter

    Arguments:
    plot_df -- df with real data and model prediction including timestep T
               from official and twitter data sources
    
    Returns:
    
    """
    t1, t2 = targets
    seasons = plot_df['season'].unique()
    
    norm_cols = targets+['twitter_original']
    normalized = plot_df[norm_cols]
    normalized = (normalized - normalized.mean())/normalized.std()
    plot_df[norm_cols] = normalized
    
    n_seasons = len(seasons)
    n_cols = 2
    n_rows = math.ceil(n_seasons / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
    axes = axes.flatten()  # for indexing
    for i, season in enumerate(seasons):
        ax = axes[i]
        group = plot_df[plot_df['season'] == season]

        ax.plot(group['wk_label'], group['official'], 'b-', label='Official', marker='o')
        ax.plot(group['wk_label'], group['twitter'], 'g-', label='Twitter', marker='x')
        ax.plot(group['wk_label'], group['twitter_original'], color='0.5', linestyle='--', label='Twitter no smoothing')

        ax.set_title(f'Flu Season {season}')
        ax.set_ylabel('Normalized ILI count')
        ax.set_ylabel('Week')
        ax.tick_params(axis='x', labelrotation=75)
        ax.legend()
        ax.grid(True)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])          # hide any unused subplots 

    plt.tight_layout()
    plt.savefig(savepath.joinpath(f"lineplot_{country}"))

