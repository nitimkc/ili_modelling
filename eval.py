#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 7 2025

@author: nitimkc

1. read data for each country
2. evaluate official and twitter time series
3. plot

"""

from pathlib import Path
import yaml 
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import eval_wrappers as ev

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')


print('creating parser')
# ======================
parser = argparse.ArgumentParser(description="ILI-Twitter Evaluation")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--data_dir", type=str, help="directory where data exists.")
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
config = CONFIG["default"]
print(f"default config")
print('\n'.join(f'    {k}: {v}' for k, v in config.items()))

# data files
# ==========
countries = {'FR':'France', 'ES':'Spain', 'IT':'Italy', 'DE':'Germany'}
targets = ['official', 'twitter']

fileids = [i for i in DATA.glob('*.csv') if config['FILE_ALIAS'] in i.name]
# geo = "it"
# fileids = [i for i in fileids if geo in i.name]
print(f"files to work with:")
print('\n'.join(f'    {item}' for item in fileids))

# for filepath in fileids:
#     country = filepath.name.split("_")[-2].upper()
#     country_config = CONFIG["country"][country]
#     print(f"\n{country} specific config")
#     print('\n'.join(f'    {k}: {v}' for k, v in country_config.items()))

#     # load data    
#     # =========
#     df = pd.read_csv(DATA.joinpath(filepath))
#     if country != 'FR':
#         df = df[df['season']>=201819]
#     df = df[(df['WK']<=17) | (df['WK']>=43)]
#     # df = df.sort_values(by=['season', 'year', 'WK'])
#     df = df.sort_values(by='week')

#     # normalize before getting description
#     # ====================================
#     norm_cols = targets
#     normalized = df[norm_cols]
#     normalized = (normalized - normalized.mean())/normalized.std()
#     normalized['twitter'] = savgol_filter(normalized['twitter'], window_length=7, polyorder=2) # savitzky-golay filter 
        
#     ev.descriptions(normalized, targets, country, 'all', SAVEPATH)
    
#     # analyze by season
#     # ===================
#     for season, season_df in df.groupby('season'):
#         print(f"\nsaving descriptions of two time series by season: {season}")
        
#         # smooth twitter series by season
#         season_df['twitter'] = savgol_filter(season_df['twitter'], window_length=7, polyorder=2) # savitzky-golay filter 
        
#         # normalize two timeseries by season
#         norm_cols = targets
#         normalized = season_df[norm_cols]
#         normalized = (normalized - normalized.mean())/normalized.std()
#         season_df[norm_cols] = normalized

#         # save descriptions by season
#         ev.descriptions(season_df, targets, country, season, SAVEPATH)
#     print("--------------------------------------")
    
#     # plot
#     # ====
#     df['twitter_original'] = df['twitter'] 
#     df['twitter'] = savgol_filter(df['twitter'], window_length=7, polyorder=2) # savitzky-golay filter 
#     # df['twitter'] = gaussian_filter1d(df['twitter'], sigma=2)
#     ev.plot_ili(df, targets, SAVEPATH, country)
    
# read results and plot
# =====================
desc_df = pd.read_json(SAVEPATH.joinpath("descriptions.json"), lines=True)
desc_df['time'] = desc_df['time'].astype(str)
desc_df['p_corr'] = desc_df['pearson_corr'].str[0]
desc_df['s_corr'] = desc_df['spearman_corr'].str[0]
desc_df = desc_df[desc_df['time']!='all']

# number of significant Granger causality results
def count_significant_granger(granger_list):
    return sum(1 for lag_dict in granger_list for val in lag_dict.values() if val[2] is True)
desc_df['granger_offtwt_sig'] = desc_df['granger_offtwt'].apply(count_significant_granger)
desc_df['granger_twtoff_sig'] = desc_df['granger_twtoff'].apply(count_significant_granger)

# total significance count across metrics
desc_df['sig_total'] = desc_df.apply(
    lambda row: sum([
        1 if row['pearson_corr'][2] else 0,
        1 if row['spearman_corr'][2] else 0,
        row['granger_offtwt_sig'],
        row['granger_twtoff_sig']
    ]), axis=1
)

def extract_lag_sig(granger_list, lag='lag1'):
    for d in granger_list:
        if lag in d:
            return 1 if d[lag][2] else 0
    return 0
desc_df['offtwt_lag1_sig'] = desc_df['granger_offtwt'].apply(lambda x: extract_lag_sig(x, 'lag1'))
desc_df['twtoff_lag1_sig'] = desc_df['granger_twtoff'].apply(lambda x: extract_lag_sig(x, 'lag1'))
desc_df['offtwt_lag2_sig'] = desc_df['granger_offtwt'].apply(lambda x: extract_lag_sig(x, 'lag2'))
desc_df['twtoff_lag2_sig'] = desc_df['granger_twtoff'].apply(lambda x: extract_lag_sig(x, 'lag2'))


# pearson corr
heatmap_pcorr = desc_df.pivot(index='country', columns='time', values='p_corr')
plt.figure(figsize=(10, 3))
sns.heatmap(heatmap_pcorr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='gray')
plt.title('Pearson Correlation per Season by Country')
plt.ylabel('Country')
plt.xlabel('ILI Season')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(SAVEPATH.joinpath(f"pearson_corr.png"))

# spearman corr
heatmap_scorr = desc_df.pivot(index='country', columns='time', values='s_corr')
plt.figure(figsize=(10, 3))
sns.heatmap(heatmap_scorr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='gray')
plt.title('Spearman Correlation per Season by Country')
plt.ylabel('Country')
plt.xlabel('ILI Season')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(SAVEPATH.joinpath(f"spearman_corr.png"))

# dtw
plt.figure(figsize=(8, 5))
sns.lineplot(data=desc_df, x='time', y='dtw_distance', hue='country', marker='o')
plt.title('DTW Distance Over Time by Country')
plt.xlabel('Time')
plt.ylabel('DTW Distance')
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVEPATH.joinpath(f"dtw.png"))

# granger1
granger_df = desc_df.melt(id_vars=['country', 'time'], 
                          value_vars=['granger_offtwt_sig', 'granger_twtoff_sig'],
                          var_name='direction', value_name='sig_count')
granger_df['direction'] = granger_df['direction'].map({
    'granger_offtwt_sig': 'Official → Twitter',
    'granger_twtoff_sig': 'Twitter → Official'
})

plt.figure(figsize=(8, 5))
sns.barplot(data=granger_df, x='country', y='sig_count', hue='direction')
plt.title('Granger Causality: # Significant Lags by Direction & Country')
plt.ylabel('Significant Lag Count')
plt.xlabel('Country')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(SAVEPATH.joinpath(f"sig.png"))

# granger2
grouped_lag1 = desc_df.groupby('country').agg(
    offtwt_total=('offtwt_lag1_sig', 'count'),
    offtwt_sig=('offtwt_lag1_sig', 'sum'),
    twtoff_sig=('twtoff_lag1_sig', 'sum')
).reset_index()

grouped_lag2 = desc_df.groupby('country').agg(
    offtwt_total=('offtwt_lag2_sig', 'count'),
    offtwt_sig=('offtwt_lag2_sig', 'sum'),
    twtoff_sig=('twtoff_lag2_sig', 'sum')
).reset_index()

# Compute normalized significance proportions
grouped_lag1['Official → Twitter'] = grouped_lag1['offtwt_sig'] / grouped_lag1['offtwt_total']
grouped_lag1['Twitter → Official'] = grouped_lag1['twtoff_sig'] / grouped_lag1['offtwt_total']
grouped_lag2['Official → Twitter'] = grouped_lag2['offtwt_sig'] / grouped_lag2['offtwt_total']
grouped_lag2['Twitter → Official'] = grouped_lag2['twtoff_sig'] / grouped_lag2['offtwt_total']

granger_lag1 = grouped_lag1.melt(
    id_vars='country',
    value_vars=['Official → Twitter', 'Twitter → Official'],
    var_name='direction',
    value_name='proportion_sig'
)
granger_lag2 = grouped_lag2.melt(
    id_vars='country',
    value_vars=['Official → Twitter', 'Twitter → Official'],
    var_name='direction',
    value_name='proportion_sig'
)
granger_lag1['lag'] = 'lag1'
granger_lag2['lag'] = 'lag2'
granger_full = pd.concat([granger_lag1, granger_lag2], ignore_index=True)

g = sns.catplot(data=granger_full, kind='bar', x='country', y='proportion_sig',
                hue='direction', col='lag', height=5, aspect=1, sharex=True)
g.set_titles("Normalized Granger Significance - {col_name}")
g.set_axis_labels("Country", "Proportion of Significant Tests")
g.set(ylim=(0, 1))
g.add_legend()
for ax in g.axes.flat:
    ax.grid(axis='y')

plt.tight_layout()
plt.savefig(SAVEPATH.joinpath("sig_granger_normalized.png"))
plt.show()