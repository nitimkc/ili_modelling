#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 7 2025

@author: nitimkc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import minimize, Parameters

def getseason(week, year, N=40):
    # influenza season
    if week>=N:
        season = f"{year}{str(year+1)[-2:]}" # beginning of the season
    else:               
        season = f"{year-1}{str(year)[-2:]}" # end of the season
        
    return season
# getseason(5,2009)

def set_params(beta, sigma, gamma):
    """
    put parameters as Parameters object

    Arguments:
    residual -- loss function

    Returns:
    solution of minimization 
    """
    params = Parameters()
    
    # Fixed grid-search parameters
    params.add('beta_s_base', value=beta, vary=False)
    params.add('sigma', value=sigma, vary=False)
    params.add('gamma_s', value=gamma, vary=False)
    
    # Other parameters (to be optimized)
    params.add('beta_s_amp', value=np.random.uniform(0, 0.5), min=0, max=2)
    params.add('beta_a_base', value=np.random.uniform(0.05, 1.5), min=0.05, max=1.5)
    params.add('beta_a_amp', value=np.random.uniform(0, 0.5), min=0, max=2)
    params.add('gamma_a', value=np.random.uniform(0.1, 0.5), min=0.1, max=0.5)
    params.add('mu', value=np.random.uniform(0, 0.1), min=0, max=0.1)
    params.add('alpha', value=np.random.uniform(0.4, 0.9), min=0.4, max=0.9)

    return params

def inflection_point(inc, t):
    """
    
    Arguments:
    
    Returns:
    
    """
    dI = np.gradient(inc, t)
    inflection_idx = np.argmax(dI)      # or detect curvature change
    inflection_point = t[inflection_idx]
    return inflection_point

def timeshift(inflection_real, inflection_pred, t):
    """
    
    Arguments:
    
    Returns:
    
    """
    delta = inflection_real - inflection_pred
    t_shifted = t + delta
    print("timeshift")
    print(t)
    print(t_shifted)
    if t_shifted[0] < 0:
        print("Warning: shifted time starts before t=0")
    return t_shifted

def plot_prediction(plot_df, filename, country):
    """
    Plot Real Data vs. SEIR Model Predictions

    Arguments:
    plot_df -- df with real data and model prediction including timestep T
               from official and twitter data sources
    
    Returns:
    save plot of model prediction against real data 
    """
    # print(plot_df.head())
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(plot_df['t'], plot_df['official'],
        'r-', label="Real Data (Influenza Cases)", 
        markersize=4, alpha=0.6)
    plt.plot(plot_df['t'], plot_df['official_pred'],
        'r--', label="Model Prediction (SEIR)", 
        linewidth=2, alpha=0.6)
    
    plt.plot(plot_df['t'], plot_df['twitter'], 
        'b-', label="Real Data (Influenza Cases) - twitter", 
        markersize=4, alpha=0.6)
    plt.plot(plot_df['t'], plot_df['twitter_pred'],
        'b--', label="Model Prediction (SEIR)", 
        linewidth=2, alpha=0.6)

    ax.set_xticks(plot_df['t'])
    ax.set_xticklabels(plot_df['wk_label'])
    plt.xticks(rotation=75)

    # plt.xlabel("Week")
    plt.ylabel(f"Number of Cases")
    plt.title(f"Real Influenza Cases vs. SEIR Model Prediction - {country}")
    plt.legend()
    plt.grid()
    plt.savefig(filename)

