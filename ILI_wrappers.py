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

from scipy.ndimage import gaussian_filter1d

def getseason(week, year, N=40):
    # influenza season
    if week>=N:
        season = f"{year}{str(year+1)[-2:]}" # beginning of the season
    else:               
        season = f"{year-1}{str(year)[-2:]}" # end of the season
        
    return season
# getseason(5,2009)

def set_params(beta, sigma, gamma, Is0_guess, Ia0_guess, E0_guess, N):
    params = Parameters()
    
    # Grid values (Fixed during the grid search stage)
    params.add('beta_s_base', value=beta,  vary=False)
    params.add('sigma',       value=sigma, vary=False)
    params.add('gamma_s',     value=gamma, vary=False)
        
    # Initial Conditions (Estimated as Unknowns)
    # By using a multiplier of the guess, the "box" adapts to each country's data scale
    # # Data-Relative Bounds (Is0_guess * 5) help solves the "Scaling Wall" problem 
    params.add('Is0',         value=Is0_guess, vary=False) 
    params.add('Ia0',         value=Ia0_guess, vary=True, min=1e-8, max=max(Ia0_guess * 3.0, 500))
    params.add('E0',          value=E0_guess,  vary=True, min=1e-8, max=max(E0_guess  * 3.0, 500))
    
    # Seasonality
    params.add('beta_s_amp',  value=np.random.uniform(0.0, 0.5),  min=0,    max=2)
    params.add('beta_a_base', value=np.random.uniform(0.05, 1.5), min=0.05, max=1.5)
    params.add('beta_a_amp',  value=np.random.uniform(0.0, 0.5),  min=0,    max=2)

    # params.add('beta_s_amp',  value=0.01, vary=True, min=0.0,  max=0.2)
    # params.add('beta_a_amp',  value=0.15, vary=True, min=0.0,  max=2.0)
    # params.add('beta_a_base', value=0.30, vary=True, min=0.01, max=2.0)
    
    # Global Parameters
    params.add('gamma_a', value=np.random.uniform(0.1, 0.5), min=0.1, max=0.5)
    params.add('mu',      value=np.random.uniform(0.0, 0.1),   min=0.0, max=0.1)
    params.add('alpha',   value=np.random.uniform(0.4, 0.9), min=0.4, max=0.9)

    # params.add('gamma_a',     value=0.33,  vary=True, min=0.1,  max=0.5)
    # params.add('mu',          value=1e-6,  vary=True, min=0.0,  max=0.1)
    # params.add('alpha',       value=0.5,   vary=True, min=0.3,  max=1.2)
    
    # Raise phi max to 1.0 to prevent peak truncation in IT/DE
    params.add('phi',         value=0.1,   vary=True, min=0.05, max=1.0) 
    
    return params

def inflection_point(inc, t, smooth=False, sigma=1.0):
    """
    Locate the first inflection point of a series

    Arguments:
    inc    : time series for which to find the inflection point
    t      : time dimension of inc
    smooth : when ``True`` inc is smoothed before derivative estimation to 
             reduce noise‑induced spurious curvature changes.
    sigma  : std dev of the Gaussian kernel used for smoothing when.
             Larger values give stronger smoothing.
    
    Returns: first inflection point occurs. If none, 0 is returned.
    """
    
    if smooth:
        inc = gaussian_filter1d(inc, sigma=sigma, mode='nearest')
        
    # changes in curvature
    d1 = np.gradient(inc, t)
    d2 = np.gradient(d1)
    sign_change = np.diff(np.sign(d2))
    inflection_idx = np.where(sign_change !=0)[0] + 1 # diff offset

    if inflection_idx.size == 0:
        return 0

    # # max curvature change
    # max_curvatures_idx = np.argmax(np.abs(d2[inflection_idx]))
    # inflection_idx = inflection_idx[max_curvatures_idx]
    
    # first curvature change
    first_inflection = inflection_idx[0]
    return t[first_inflection]

def timeshift(inflection_real, inflection_pred, t):
    
    """
    Shifts the model's time array so that its inflection point aligns with the observed one.

    Parameters:
        inflection_real: float - inflection point of observed data
        inflection_pred: float - inflection point of model prediction
        t: np.array - original model time array

    Returns:
        t_shifted: np.array - model time array shifted so inflection aligns
    """

    delta = inflection_real - inflection_pred
    print(f"    timeshift calibration by {delta:.2f}.")
 
    t_shifted = t + delta
    if t_shifted[0] < 0:
        print("    Warning: Model detects changes in disease spread after they appear in real data.")
    print("timeshift")
    print(t)
    print(t_shifted)
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
        color='#ff7f0e', linestyle='-', label="Real Data (Influenza Cases)", 
        markersize=4, alpha=0.6)
    plt.plot(plot_df['t'], plot_df['official_pred'],
        color='#ff7f0e', linestyle='--', label="Model Prediction (SEIR)", 
        linewidth=2, alpha=0.6)
    
    plt.plot(plot_df['t'], plot_df['classified'], 
        color='#6cbdf5', linestyle='-', label="Real Data (Influenza Cases) - twitter", 
        markersize=4, alpha=0.3)  
    plt.plot(plot_df['t'], plot_df['classified_smoothed'], 
        color='#1f77b4', linestyle='-', label="Real Data (Influenza Cases) - twitter smoothed", 
        markersize=4, alpha=0.6)  
    plt.plot(plot_df['t'], plot_df['classified_pred'],
        color='#1f77b4', linestyle='--', label="Model Prediction (SEIR)", 
        linewidth=2, alpha=0.6)

    ax.set_xticks(plot_df['t'])
    ax.set_xticklabels(plot_df['wk_label'])
    plt.xticks(rotation=75)

    # plt.xlabel("Week")
    plt.ylabel(f"Number of Cases")
    plt.title(f"Real Influenza Cases vs. SEIR Model Prediction - {country}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename)

def plot_all_countries_grid(countries, save_dir, grid_filename="SEIR_Final_Grid.png"):
    """
    Reloads saved normalized data and plots a 2x2 comparison grid with a single shared legend.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 13)) # Increased height slightly for legend
    axes = axes.flatten()

    for i, country in enumerate(countries):
        ax = axes[i]
        filepath = save_dir.joinpath(f"plot_normalized_{country}.csv")
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found.")
            continue
            
        df = pd.read_csv(filepath)
        
        # Plot Official (Orange)
        ax.plot(df['t'], df['official'], color='#ff7f0e', linestyle='-', 
                label="Official Real", alpha=0.4)
        ax.plot(df['t'], df['official_pred'], color='#ff7f0e', linestyle='--', 
                label="Official Pred (Erlang)", linewidth=2)
        
        # Plot Twitter (Blue)
        ax.plot(df['t'], df['classified_smoothed'], color='#1f77b4', linestyle='-', 
                label="Twitter Smooth", alpha=0.4)
        ax.plot(df['t'], df['classified_pred'], color='#1f77b4', linestyle='--', 
                label="Twitter Pred (Erlang)", linewidth=2)

        ax.set_title(f"Model Fit: {country}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Normalized Intensity")
        
        # Clean up x-axis
        step = 3
        ax.set_xticks(df['t'][::step])
        ax.set_xticklabels(df['wk_label'][::step], rotation=45, fontsize=9)
        ax.grid(alpha=0.2)

    # 1. Extract labels and handles from any plot (since they are identical)
    handles, labels = axes[0].get_legend_handles_labels()

    # 2. Adjust layout to make room at the bottom for the legend
    plt.tight_layout(rect=[0, 0.08, 1, 1]) 

    # 3. Create a single figure-level legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.05))
    
    plt.savefig(save_dir.joinpath(grid_filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Final 2x2 comparison grid saved to {grid_filename}")
