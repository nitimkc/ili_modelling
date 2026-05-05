#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:53:22 2025

@author: llopez, nitimkc
"""

import logging
import sys
from tqdm import tqdm

import time
import json

import random
from itertools import product
import numpy as np
import pandas as pd

from scipy.integrate import odeint
from scipy.signal import savgol_filter
from lmfit import minimize, Parameter, report_fit, Model, Minimizer

from ILI_wrappers import set_params, inflection_point, timeshift

from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Output to stdout
    )
logger = logging.getLogger()

def extract_param_values(params):
    """Extract just the .value of each lmfit Parameter into a flat dict."""
    return {k: v.value for k, v in params.items()}

def fit_seir(df, target, label, N, config, savepath, time_calibration=True):
    # fit seir_expo_model to data


    # 1. Initial conditions
    # ======================
    print(f"    Setting initial conditions")
    I_real = df[target].values # real incidences
    t_data = df['t'].values    # time stamp
    T = df.shape[0] 

    Is0 = I_real[0]    # Initial symptomatic
    Ia0 = Is0 * 1.0
    E_total_0 = Is0 * 1.0  # Total starting Exposed
        
    # Distribute E_total_0 across the 3 sub-compartments
    E1_0 = E_total_0 / 3
    E2_0 = E_total_0 / 3
    E3_0 = E_total_0 / 3
    
    R0, D0 = 0, 0
    S0 = N - (E1_0 + E2_0 + E3_0 + Is0 + Ia0 + R0 + D0)
    
    # New y0 with 8 elements
    y0 = [S0, E1_0, E2_0, E3_0, Is0, Ia0, R0, D0]
    print(f"Initial conditions: {y0}")
    assert abs(sum(y0) - N) < 1e-6, "Initial compartments don’t sum to N"
    
    # # solve the system over T*7 days
    # t_span = (min(t_data), max(t_data))          
    # t_eval = np.linspace(*t_span, max(t_data))

    # 2. Build params grid to search over
    # ===================================
    beta_s_base = np.linspace(config["beta_s_base_low"], config["beta_s_base_high"], config["beta_s_base_ntry"])
    sigma = np.linspace(config["sigma_low"], config["sigma_high"], config["sigma_ntry"])
    gamma_s = np.linspace(config["gamma_s_low"], config["gamma_s_high"], config["gamma_s_ntry"])
    param_grid = list(product(beta_s_base, sigma, gamma_s))
    if config["n_sample"]>0:
        param_grid = random.sample(param_grid, config["n_sample"])
    print(f"    Number of fits in parameter grid: {len(param_grid)}")
    

    # 3. Fit over grid
    # ================
    start_time = time.time()

    # in loop
    # try:
    #     np.random.seed(42)
    #     results = []
    #     for i, (beta, sigma, gamma) in enumerate(tqdm(param_grid, desc="Fitting models")):
    #         params = set_params(beta, sigma, gamma)
    #         fit_start = time.time()
    #         result = optimize(residual, params, (t_data, y0, I_real), 
    #                           label, target, savepath)
    #         fit_duration = time.time() - fit_start
    #         if (i + 1) % 10 == 0:
    #             tqdm.write(
    #                 f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Fit {i + 1}: "
    #                 f"beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, "
    #                 f"residual={result.chisqr:.4f}, duration={fit_duration:.2f}s"
    #             )
    #         results.append(result)
    # except Exception as e:
    #     logger.warning(f"Fit {i + 1} failed: {e}")
    
    # in parallel 
    np.random.seed(42)
    results = Parallel(n_jobs=-1)(
        delayed(run_fit)(
            i, beta, sigma, gamma, t_data, y0, I_real, label, target, savepath
        )
        for i, (beta, sigma, gamma) in enumerate(tqdm(param_grid))
    )
    
    results = [r for r in results if r is not None]
    total_duration = time.time() - start_time
    logger.info(f"All fits completed in {total_duration:.2f} seconds")


    # 4. Find best result
    # ===================
    if results:
        best_result = min(results, key=lambda r: r.chisqr)
        best_params = best_result.params
        print(f"    Best fit:") 
        best_params.pretty_print()
        # logger.info(
        #     f"Best fit: beta={best_params['beta_s_base'].value:.3f}, "
        #     f"sigma={best_params['sigma'].value:.3f}, "
        #     f"gamma={best_params['gamma_s'].value:.3f}, "
        #     f"residual={best_result.chisqr:.4f}"
        # )
        best_param_dict = {k: v.value for k, v in best_params.items()}
        best_param_dict['country'] = label
        best_param_dict['target'] = target
        with open(savepath.joinpath(f"seir_best_params.json"), "a") as f:
            json.dump(best_param_dict, f, indent=4)
    else:
        logger.error("No successful fits found.")
        

    # 5. Model based on best result
    # =============================
    print(f"    Simulating SEIR with best params:")
    # y0 - change to the other time series
    I_pred = simulate_SEIR(y0, t_data, label, target, best_params, savepath, "best")
    
    # can smmooth before getting inflection point of real data
    # I_real_smoothed = smooth_curve(I_real)
    # I_real_smoothed = savgol_filter(I_real, window_length=7, polyorder=2)
    

    # 6. Time-shift calibration of I_model
    # ====================================
    if time_calibration:
        t_inflect_real = inflection_point(I_real, t_data, smooth=True) # of obs
        t_inflect_pred = inflection_point(I_pred, t_data, smooth=True) # of pred 
        print(f"    Real and predicted inflection point: {t_inflect_real}, {t_inflect_pred}")

        if t_inflect_real != t_inflect_pred:
            try:
                t_shifted = timeshift(t_inflect_real, t_inflect_pred, t_data)
                t_shifted_result = optimize(residual, best_params, (t_shifted, y0, I_real),
                                            savepath, "timeshifted")
                t_shifted_params = t_shifted_result.params                               
                I_pred = simulate_SEIR(y0, t_shifted, label, target, t_shifted_params, savepath, "t_shifted")
                I_pred = I_pred[:T]
            
            except ValueError as e:
                print(f"Time calibration skipped: {e}")
        tqdm.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - No time shift needed "
            f"(inflection point aligned at t = {t_inflect_real:.2f})"
                )
    else:
        tqdm.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Time Calibration = FALSE "
                )

    return I_pred[:T]   

def run_fit(i, beta, sigma, gamma, t_data, y0, I_real, label, target, savepath):
    try:
        params = set_params(beta, sigma, gamma)

        fit_start = time.time()
        result = optimize(residual, params, (t_data, y0, I_real), 
                          label, target, savepath)
        fit_duration = time.time() - fit_start
        if (i + 1) % 10 == 0:
            tqdm.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Fit {i + 1}: "
                f"beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, "
                f"residual={result.chisqr:.4f}, duration={fit_duration:.2f}s"
            )
        
        return result

    except Exception as e:
        tqdm.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - Fit {i + 1} failed: {e}")
        return None

def optimize(residual, params, args, label, target, savepath=None, filename=""):
    """
    Fit SEIR curve

    Arguments:
    residual -- loss function
    params -- 
    args -- (t, y0, I_real)

    Returns:
    solution of minimization 
    """
    fit_start = time.time()
    result = minimize(residual, params, nan_policy='omit',
                      args=args, method='least_squares',
                      max_nfev=10000, ftol=1e-8, gtol=1e-8, xtol=1e-8, loss='huber',
                      # max_nfev=100000, ftol=1e-10, gtol=1e-10, xtol=1e-10, loss='arctan', 
                      diff_step=1e-4, tr_solver='lsmr', 
                      verbose=0)

    # logger.info(f"Fit {i + 1}: beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, residual={result.chisqr:.4f}")
    if savepath is not None:
        save_result = {
            'country': label,
            'target': target,
            'beta': params['beta_s_base'].value,
            'sigma': params['sigma'].value,
            'gamma': params['gamma_s'].value,
            'residual': result.chisqr,
            'opt_params': extract_param_values(result.params) # nested dict
            }
        json_fitpath = savepath.joinpath(f"seir_{filename}fit_results.json")
        with open(json_fitpath, "a") as f:
            f.write(json.dumps(save_result) + ",\n")

    return result

def simulate_SEIR(y0, t_data, label, target, params, savepath=None, filename=""):
    """
    simulate SEIR with given params 

    Arguments:
    results -- solution for given set of parameters

    Returns:
    solution of minimization 
    """
    # Simulate SEIR model
    soln = odeint(seir_expo_model, y0, t_data, args=(params,))
    soln_df = pd.DataFrame(soln, columns=['S', 'E', 'Is', 'Ia', 'R', 'D'])
    soln_df['country'] = label
    soln_df['target'] = target
    soln_df['t'] = t_data 

    if savepath:
        json_bestfitpath = savepath.joinpath(f"seir_{filename}_solution.json")
        with open(json_bestfitpath, "a") as f:
            soln_df.to_json(f, orient='records', lines=True)
            f.write("\n")

    Is_pred = soln_df['Is'] # predicted Symptomatic Infected (Is) from the SEIR simulation
    return Is_pred

def seir_expo_model(y, t, params):
    
    S, E1, E2, E3, Is, Ia, R, D = y
    m = 3 # Number of sub-compartments
    
     # Extract parameters from lmfit
    beta_s_base = params['beta_s_base'].value  # Base transmission rate (symptomatic)
    beta_s_amp = params['beta_s_amp'].value    # Amplitude of seasonal variation (symptomatic)
    beta_a_base = params['beta_a_base'].value  # Base transmission rate (asymptomatic)
    beta_a_amp = params['beta_a_amp'].value    # Amplitude of seasonal variation (asymptomatic)
    
    sigma = params['sigma'].value             # Incubation rate (E -> I)
    gamma_s = params['gamma_s'].value         # Recovery rate (symptomatic)
    gamma_a = params['gamma_a'].value         # Recovery rate (asymptomatic)
    mu = params['mu'].value                   # Mortality rate (symptomatic)
    alpha = params['alpha'].value             # Proportion of symptomatic infections

    N = S + E1 + E2 + E3 + Is + Ia + R + D    # Total population
    
    # Time-dependent transmission rates with sinusoidal variation
    omega = 2 * np.pi / (30*4)  # Frequency corresponding to a 6-month period
    beta_s = beta_s_base * (1 + beta_s_amp * np.sin(omega * t))
    beta_a = beta_a_base * (1 + beta_a_amp * np.sin(omega * t))
    
    # Effective transition rate through EACH sub-stage
    # Total time 1/sigma is split across m stages
    k = m * sigma 

    # Differential equations
    dS = - (beta_s * Is + beta_a * Ia) * S / N
    
    # Erlang sub-compartment chain
    dE1 = (beta_s * Is + beta_a * Ia) * S / N - k * E1
    dE2 = k * E1 - k * E2
    dE3 = k * E2 - k * E3
    
    # Progress to Infectious from the FINAL sub-compartment (E3)
    dIs = alpha * (k * E3) - (gamma_s + mu) * Is
    dIa = (1 - alpha) * (k * E3) - gamma_a * Ia  
    dR = gamma_s * Is + gamma_a * Ia
    dD = mu * Is
    
    return [dS, dE1, dE2, dE3, dIs, dIa, dR, dD]

def g(t, y0, params):
    return odeint(seir_expo_model, y0, t, args=(params,))

def residual(param, t, y0, I_real, N):
    """
    Objective function for fitting the SEIR model to real data.
    """
    # y0 = [S0, E0, Is0, Ia0, R0, D0]  # Initial conditions
    C_real = I_real.cumsum()                               # cumulative incidences

    sol =  g(t, y0, param)
    # sol = odeint(seir_expo_model, y0, t_data, args=(params,))
    
    I_model = sol[:, 2]  # Symptomatic Infected (Is)
    C_model = np.cumsum(I_model)  # Cumulative reported cases
    
    Inf = ((I_model - I_real) / np.max(I_real)).ravel()
    Cum = ((C_model - C_real) / np.max(C_real)).ravel()
    
    resid = np.array([Inf * 5, Cum * 3])
    # resid = np.array([Inf, Cum])
    return resid

