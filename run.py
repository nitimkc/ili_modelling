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

# Define model parameters using lmfit
# TO DO - move to config
params = Parameters()
params.add('beta_s_base', value=np.random.uniform(0.1, 0.5), min=0.1, max=5)  # Base transmission rate (symptomatic)
params.add('beta_s_amp', value=np.random.uniform(0, 0.5), min=0, max=2)  # Seasonal variation amplitude (symptomatic)
params.add('beta_a_base', value=np.random.uniform(0.05, 1.5), min=0.05, max=1.5)  # Base transmission rate (asymptomatic)
params.add('beta_a_amp', value=np.random.uniform(0, 0.5), min=0, max=2)  # Seasonal variation amplitude (asymptomatic)

params.add('sigma', value=np.random.uniform(1/5, 1/2), min=1/5, max=1/2)   # Incubation rate (E → I)
params.add('gamma_s', value=np.random.uniform(1/14, 1/3), min=1/14, max=1/3) # Recovery rate (symptomatic)
params.add('gamma_a', value=np.random.uniform(1/10, 1/3), min=1/10, max=1/3) # Recovery rate (asymptomatic)
params.add('mu', value=np.random.uniform(0, 0.1), min=0, max=0.1)   # Mortality rate (symptomatic)
params.add('alpha', value=np.random.uniform(0.4, 0.9), min=0.4, max=0.9)   # Proportion of symptomatic infections

countries = {'FR':'France', 'ES':'Spain', 'IT':'Italy', 'DE':'Germany'}
fileids = [i for i in DATA.glob('*.csv') if CONFIG['FILE_ALIAS'] in i.name]
# print(fileids)

for filepath in fileids:
    country = filepath.name.split("_")[-2].upper()
    print(country)

    # prepare data to fit
    df_influenza = pd.read_csv(DATA.joinpath(filepath))
    df_fit = df_influenza[(df_influenza['season'] == CONFIG['SEASON'])] 
    df_fit = df_fit[(df_fit['week'] >= CONFIG['START_WEEK'])]          # take flu-season weeks only

    df_fit.sort_values(by='week', inplace=True)
    df_fit['t'] = np.arange(0, len(df_fit)) * 7
    df_fit = df_fit[df_fit['t']<=24*7]  

    targets = ['official', 'twitter']
    T = df_fit.shape[0]

    # data to plot
    plot_df = df_fit[['t','wk_label']+targets]
    for target in targets:
        print(target)

        # fit model
        sol = fit_seir(df_fit, target, params)
        S, E, Is, Ia, R, D = sol[:,0],sol[:,1],sol[:,2],sol[:,3],sol[:,4],sol[:,5] #sol.y

        # model predictions from the SEIR simulation
        I_model = Is      # Extract Symptomatic Infected (Is) from the model
        plot_df[f"{target}_pred"] =  I_model[:T]
    # print(plot_df.shape)

    # normalize
    norm_indicator = None
    if CONFIG['NORMALIZE']:
        cols = [i for i in plot_df.columns if i!='t']
        normalized = plot_df[cols]
        normalized = (normalized - normalized.mean())/normalized.std()
        plot_df = pd.concat([plot_df[['t']], normalized], axis=1)
        norm_indicator = 'normalized'
    
    # Plot Real Data vs. Model Predictions
    filename = f"SEIR_{country}_{norm_indicator}predplot.png"
    plot_prediction(plot_df, 
                    SAVEPATH.joinpath(filename), 
                    countries[country]
                    )
    print(f"{filename} saved")
    print("=================")