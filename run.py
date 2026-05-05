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
from ILI_wrappers import plot_prediction, plot_all_countries_grid

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
df = pd.read_csv(DATA.joinpath("combined_ili_season_apr2026.csv"))

df['week_str'] = df['week'].astype(str)
df['year'] = df['week_str'].str[:4].astype(int)
df['WK'] = df['week_str'].str[4:].astype(int)
df['wk_label'] = 'WK-' + df['week_str'].str[4:]

# loop over country
df_countries = ["FR", "IT", "ES", "DE"]#set(df["country"].unique())
for country in df_countries:
    if country not in CONFIG["country"]:
        print(f"Skipping {country}: No configuration found in YAML.")
        continue

    # access country config
    country_config = CONFIG["country"][country]
    print(f"\nProcessing {country} specific config...")
    print('\n'.join(f'    {k}: {v}' for k, v in country_config.items()))
    
    targets = config['TARGETS']
    info = ['t','wk_label']
    N = country_config['popn'] # popn: https://ec.europa.eu/eurostat/documents/2995521/9063738/3-10072018-BP-EN.pdf/ccdfc838-d909-4fd8-b3f9-db0d65ea457f#:~:text=EU%20population%20up%20to%20nearly,Population%20Day%20(11%20July).
    # https://worldpopulationreview.com/country-rankings/twitter-users-by-country
    
    # prepare data to fit    
    # ===================
    df_influenza = df[df['country'] == country].copy()                         # country to analyze
    df_fit = df_influenza[(df_influenza['season'] == config['SEASON'])]        # season to analyze
    df_fit = df_fit[(df_fit['week'] >= country_config['start_week'])]          # take flu-season weeks only

    df_fit.sort_values(by='week', inplace=True)                                
    if country == 'DE':
        holiday_weeks = [201852, 201901]        
        df_fit.loc[df_fit['week'].isin(holiday_weeks), 'official'] = np.nan
        df_fit['official'] = df_fit['official'].interpolate(method='linear')   # average of wk-51 and wk-01    
        print(f"    Bridged DE holiday reporting trough (Weeks 52 and 01)")

    df_fit['official'] = (df_fit['official'] / 100_000) * N                    # absolute value rather than incidence
    df_fit['t'] = np.arange(0, len(df_fit)) * 7                                # add time point   
    df_fit = df_fit[df_fit['t']<=30*7]  
    
    # out dir
    any_timeshift = country_config.get('time_calibration_official') or country_config.get('time_calibration_classified')
    ts_label = "timeshift" if any_timeshift else "no-shift"
    folder_name = f"{ts_label}"
    SAVEPATH = ROOT / args.out_dir / folder_name
    SAVEPATH.mkdir(parents=True, exist_ok=True)
    # print(SAVEPATH)

    plot_df = df_fit[info+targets] # data to plot
    plot_df = plot_df.reset_index(drop=True)
    print(plot_df.head())

    inflection_points = {}
    for target in targets:
        print(f"\n    Fitting {target}")
        # ============================
        
        if target != 'official': 
            print("    Smoothing ILI-Tweets")   
            
            df_fit[target] = savgol_filter(df_fit[target].values, window_length=5, polyorder=2) # savitzky-golay filter 
            # df_fit[target] = gaussian_filter1d(df_fit[target], sigma=2) # gaussian filter
            
            plot_df[f"{target}_smoothed"] = df_fit[target].values
        
        timeshift = True if country_config[f'time_calibration_{target}'] else False
        pred, t_inflect = fit_seir(df_fit, target, country, N, config['PARAMS'], SAVEPATH, time_calibration=timeshift)
        inflection_points[target] = t_inflect

        if pred is not None:
            plot_df[f"{target}_pred"] =  pred # fit model
        else:
            plot_df[f"{target}_pred"] =  None
            print("Predictions are None")
    print(plot_df.head())
    plot_df.to_csv(SAVEPATH.joinpath(f"plot_{country}.csv"), index=False)

    # Lead time analysis
    # ==================
    if 'official' in inflection_points and 'classified' in inflection_points:
        # Lead time = Official time - Twitter time
        # Positive value means Twitter is earlier (leads)
        lead_time = inflection_points['official'] - inflection_points['classified']
        
        print(f"\n[LEAD TIME ANALYSIS - {country}]")
        print(f"    Official Inflection: {inflection_points['official']:.2f} days")
        print(f"    Twitter Inflection:  {inflection_points['classified']:.2f} days")
        print(f"    Twitter Leads by:    {lead_time:.2f} days")

        # Save result to a text file for your thesis records
        with open(SAVEPATH.joinpath(f"lead_time_{country}.txt"), "w") as f:
            f.write(f"Country: {country}\n")
            f.write(f"Official Inflection (days): {inflection_points['official']}\n")
            f.write(f"Twitter Inflection (days): {inflection_points['classified']}\n")
            f.write(f"Lead Time (days): {lead_time}\n")

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
        filename = f"SEIR_{country}_{norm_indicator}predplot_popn.png"
    else:
        norm_indicator = ''
        filename = f"SEIR_{country}_{norm_indicator}predplot_popn.png"
    
    
    # Plot real vs predictions
    # ========================
    plot_prediction(plot_df, 
                    SAVEPATH.joinpath(filename), 
                    country
                    )
    print(f"{filename} saved")

# all countries in one grid
plot_all_countries_grid(df_countries, SAVEPATH)
