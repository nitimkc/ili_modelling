#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 7 2025

@author: nitimkc

1. read data for each country
2. create SEIR model using official and twitter data
3. plot

"""

from pathlib import Path
import yaml 
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from ILI_SEIR import fit_seir
from ILI_wrappers import plot_prediction

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')


print('creating parser')
# ======================
parser = argparse.ArgumentParser(description="SEIR ILI-Twitter Evaluation")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--data_dir", type=str, help="directory where images exists.")
parser.add_argument("--out_dir", type=str, help="directory where results are to be saved.")
parser.add_argument("--config_file", type=str, help="file containing run configurations.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(args.data_dir)
CONFIG_FILE = ROOT.joinpath(args.config_file)

with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)
config = CONFIG["default"]
print(f"default config")
print('\n'.join(f'    {k}: {v}' for k, v in config.items()))


# data files
# ==========
countries = {'FR':'France', 'ES':'Spain', 'IT':'Italy', 'DE':'Germany'}
fileids = [i for i in DATA.glob('*.csv') if config['FILE_ALIAS'] in i.name]
# geo = "it"
# fileids = [i for i in fileids if geo in i.name]
print(f"files to work with:")
print('\n'.join(f'    {item}' for item in fileids))

for filepath in fileids:
    country = filepath.name.split("_")[-2].upper()
    country_config = CONFIG["country"][country]
    print(f"\n{country} specific config")
    print('\n'.join(f'    {k}: {v}' for k, v in country_config.items()))
    
    targets = ['official', 'twitter']
    info = ['t','wk_label']
    N = country_config['popn'] # popn: https://ec.europa.eu/eurostat/documents/2995521/9063738/3-10072018-BP-EN.pdf/ccdfc838-d909-4fd8-b3f9-db0d65ea457f#:~:text=EU%20population%20up%20to%20nearly,Population%20Day%20(11%20July).

    # prepare data to fit    
    # ===================
    df_influenza = pd.read_csv(DATA.joinpath(filepath))
    # df_influenza['official'] = df_influenza['official']/N
    df_fit = df_influenza[(df_influenza['season'] == config['SEASON'])] 
    df_fit = df_fit[(df_fit['week'] >= country_config['start_week'])]          # take flu-season weeks only

    df_fit.sort_values(by='week', inplace=True)
    df_fit['t'] = np.arange(0, len(df_fit)) * 7
    df_fit = df_fit[df_fit['t']<=24*7]  
    
    df_fit['official'] = (df_fit['official'] / 100_000) * N
    popn_norm_indicator = "adj"
    SAVEPATH = ROOT.joinpath(f"{args.out_dir}_popn{popn_norm_indicator}")
    SAVEPATH.mkdir(parents=True, exist_ok=True)

    plot_df = df_fit[info+targets] # data to plot
    plot_df = plot_df.reset_index(drop=True)
    print(plot_df.head())

    for target in targets:
        print(f"    Fitting {target}")
        # ============================
        
        if target == 'twitter': 
            print("    Smoothing twitter ILI incidences")   
            df_fit[target] = savgol_filter(df_fit[target].values, window_length=5, polyorder=2) # savitzky-golay filter 
            # df_fit[target] = gaussian_filter1d(df_fit[target], sigma=2) # gaussian filter
            plot_df[f"{target}_smoothed"] = df_fit[target].values
        timeshift = True if country_config[f'time_calibration_{target}'] else False
        pred = fit_seir(df_fit, target, country, N, config['PARAMS'], SAVEPATH, 
                        time_calibration=timeshift)
        if pred is not None:
            plot_df[f"{target}_pred"] =  pred # fit model
        else:
            plot_df[f"{target}_pred"] =  None
            print("Predictions are None")
    print(plot_df.head())
    plot_df.to_csv(SAVEPATH.joinpath(f"plot_{country}.csv"), index=False)


    # normalize
    # =========
        
    if config['PLOT_NORMALIZE']:
        norm_cols = [i for i in plot_df.columns if i not in info]
        normalized = plot_df[norm_cols]
        
        # normalized = (normalized - normalized.mean())/normalized.std()
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        plot_df = pd.concat([plot_df[info], normalized], axis=1)
        plot_df.to_csv(SAVEPATH.joinpath(f"plot_normalized_{country}.csv"), index=False)
        
        norm_indicator = 'normalized'
        filename = f"SEIR_{country}_{norm_indicator}predplot_popn{popn_norm_indicator}.png"
    else:
        norm_indicator = ''
        filename = f"SEIR_{country}_{norm_indicator}predplot_popn{popn_norm_indicator}.png"
    
    # Plot real vs predictions
    # ========================
    plot_prediction(plot_df, 
                    SAVEPATH.joinpath(filename), 
                    countries[country]
                    )
    print(f"{filename} saved")