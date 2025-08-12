#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 7 2025

@author: nitimkc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import minimize
from ILI_SEIR import residual

def getseason(week, year, N=40):
    # influenza season
    if week>=N:
        season = f"{year}{str(year+1)[-2:]}" # beginning of the season
    else:               
        season = f"{year-1}{str(year)[-2:]}" # end of the season
        
    return season
# getseason(5,2009)


def plot_prediction(plot_df, filename, country):
    """
    Plot Real Data vs. SEIR Model Predictions

    Arguments:
    plot_df -- df with real data and model prediction including timestep T
               from official and twitter data sources
    
    Returns:
    save plot of model prediction against real data 
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(plot_df['t'], plot_df['official'],
        'r-', label="Real Data (Influenza Cases)", markersize=4)
    plt.plot(plot_df['t'], plot_df['official_pred'],
        'r--', label="Model Prediction (SEIR)", linewidth=2)
    
    plt.plot(plot_df['t'], plot_df['twitter'], 
        'b-', label="Real Data (Influenza Cases) - twitter", markersize=4)
    plt.plot(plot_df['t'], plot_df['twitter_pred'],
        'b--', label="Model Prediction (SEIR)", linewidth=2)
    ax.set_xticks(plot_df['t'])
    ax.set_xticklabels(plot_df['wk_label'])
    plt.xticks(rotation=75)

    # plt.xlabel("Week")
    plt.ylabel(f"Number of Cases")
    plt.title(f"Real Influenza Cases vs. SEIR Model Prediction - {country}")
    plt.legend()
    plt.grid()
    plt.savefig(filename)

