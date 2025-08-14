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
from ILI_SEIR import fit_seir

from ILI_wrappers import plot_prediction

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

print('creating parser')
parser = argparse.ArgumentParser(description="SEIR ILI-Twitter Evaluation")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--data_dir", type=str, help="directory where images exists.")
parser.add_argument("--out_dir", type=str, help="directory where results are to be saved.")
parser.add_argument("--config_file", type=str, help="file containing run configurations.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(args.data_dir)
CONFIG_FILE = ROOT.joinpath(args.config_file)
SAVEPATH = ROOT.joinpath(args.out_dir)
SAVEPATH.mkdir(parents=True, exist_ok=True)
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)
CONFIG = CONFIG["default"]
# print(CONFIG)

# data files
countries = {'FR':'France', 'ES':'Spain', 'IT':'Italy', 'DE':'Germany'}
# https://ec.europa.eu/eurostat/documents/2995521/9063738/3-10072018-BP-EN.pdf/ccdfc838-d909-4fd8-b3f9-db0d65ea457f#:~:text=EU%20population%20up%20to%20nearly,Population%20Day%20(11%20July).
popn = {'FR':67.2e6, 'ES':46.7e6, 'IT':60.5e6, 'DE':82.9e6} 
fileids = [i for i in DATA.glob('*.csv') if CONFIG['FILE_ALIAS'] in i.name]
for filepath in fileids:
    country = filepath.name.split("_")[-2].upper()

    # prepare data to fit
    df_influenza = pd.read_csv(DATA.joinpath(filepath))
    df_fit = df_influenza[(df_influenza['season'] == CONFIG['SEASON'])] 
    df_fit = df_fit[(df_fit['week'] >= CONFIG['START_WEEK'])]          # take flu-season weeks only

    df_fit.sort_values(by='week', inplace=True)
    df_fit['t'] = np.arange(0, len(df_fit)) * 7
    df_fit = df_fit[df_fit['t']<=24*7]  

    targets = ['official', 'twitter']
    info = ['t','wk_label']
    T = df_fit.shape[0]
    plot_df = df_fit[info+targets] # data to plot
    
    for target in targets:
        print(f"SEIR for {country} {target}")
        pred = fit_seir(df_fit, target, popn[country], country, SAVEPATH)
        print(pred)
        if pred is not None:
            plot_df[f"{target}_pred"] =  pred # fit model
    # print(plot_df.shape)

    # normalize
    if CONFIG['NORMALIZE']:
        norm_cols = [i for i in plot_df.columns if i not in info]
        normalized = plot_df[norm_cols]
        normalized = (normalized - normalized.mean())/normalized.std()
        plot_df = pd.concat([plot_df[info], normalized], axis=1)
    
    # Plot Real Data vs. Model Predictions
    norm_indicator = 'normalized' if CONFIG['NORMALIZE'] else None
    filename = f"SEIR_{country}_{norm_indicator}predplot_popnadj.png"
    plot_prediction(plot_df, 
                    SAVEPATH.joinpath(filename), 
                    countries[country]
                    )
    print(f"{filename} saved")
    print("=================")