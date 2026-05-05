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

def fit_seir(df, target, label, N, config, savepath, time_calibration=True, erlang=True):
    # fit seir_model to data

    # 1. Initial conditions
    # ======================
    print(f"    Setting initial conditions")
    I_real = df[target].values # real incidences
    t_data = df['t'].values    # time stamp
    T = df.shape[0] 
    
    Is0 = I_real[0]    # Initial symptomatic
    Ia0 = Is0 * 2.0
    E0  = Is0 * 2.0    # Initial total exposed but not infectious individuals
    R0  = 0
    D0  = 0            # Death
    
    S0 = N - E0 - Ia0 - Is0 - R0 - D0
        
    if erlang:
        y0 = [S0, E0, 0, 0, Is0, Ia0, R0, D0]
    else:
        y0 = [S0, E0_total, Is0, Ia0, R0, D0]
    
    print(f"Initial conditions: {y0}")
    assert abs(sum(y0) - N) < 1e-6, "Initial compartments don’t sum to N"


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
    
    # in parallel 
    results = Parallel(n_jobs=32,  backend="multiprocessing", batch_size=25, verbose=0)(
        delayed(run_fit)( 
            i, beta, sigma, gamma, t_data, y0, I_real, label, target, N, savepath
        )
        for i, (beta, sigma, gamma) in enumerate(param_grid)
    )
    
    valid_results = [r for r in results if r is not None and hasattr(r, 'save_data')]
    if valid_results:
        json_fitpath = savepath.joinpath(f"seir_fit_results.json")
        with open(json_fitpath, "a") as f:
            for r in valid_results:
                f.write(json.dumps(r.save_data) + ",\n")

    total_duration = time.time() - start_time
    logger.info(f"All fits completed in {total_duration:.2f} seconds")


    # 4. Find best result
    # ===================
    if results:
        best_result = min(results, key=lambda r: r.chisqr)
        best_params = best_result.params
        print(f"    Best fit:") 
        best_params.pretty_print()
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
    I_pred = simulate_SEIR(t_data, label, target, best_params, N, savepath, "best")
    

    # 6. Time-shift calibration of I_model
    # ====================================
    natural_inflect = inflection_point(I_pred, t_data, smooth=True)    # of pred

    if time_calibration:
        t_inflect_real = inflection_point(I_real, t_data, smooth=True) # of obs
        t_inflect_pred =  t_inflect_pred = natural_inflect
        print(f"    Real and predicted inflection point: {t_inflect_real}, {t_inflect_pred}")

        if t_inflect_real != t_inflect_pred:
            try:
                t_shifted = timeshift(t_inflect_real, t_inflect_pred, t_data)
                t_shifted_result = optimize(residual, best_params, (t_shifted, I_real, N),
                                            label, target, savepath, "timeshifted")
                t_shifted_params = t_shifted_result.params                               
                I_pred = simulate_SEIR(t_shifted, label, target, t_shifted_params, N, savepath, "t_shifted")
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
    
    return I_pred[:T], natural_inflect  

def run_fit(i, beta, sigma, gamma, t_data, y0, I_real, label, target, N, savepath):
    
    np.random.seed(None) 
    
    try:
        params = set_params(beta, sigma, gamma, y0[4], y0[5], y0[1], N)

        fit_start = time.time()
        result = optimize(residual, params, (t_data, I_real, N), label, target, savepath, precision="fast")
        fit_duration = time.time() - fit_start
        
        result.save_data = {
            'country': label, 
            'target': target,
            'grid_beta': beta, 
            'grid_sigma': sigma, 
            'grid_gamma': gamma,
            'residual': float(result.chisqr),
            'opt_params': {k: v.value for k, v in result.params.items()}
        }
        # if (i + 1) % 10 == 0:
        #     tqdm.write(
        #         f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Fit {i + 1}: "
        #         f"beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, "
        #         f"residual={result.chisqr:.4f}, duration={fit_duration:.2f}s"
        #     )
        
        return result

    except Exception as e:
        tqdm.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - Fit {i + 1} failed: {e}")
        return None

def optimize(residual, params, args, label, target, savepath=None, filename="",  precision="fast"):
    """
    Fit SEIR curve

    Arguments:
    residual -- loss function
    params -- 
    args -- (t, y0, I_real)

    Returns:
    solution of minimization 
    """
    tol = 1e-4 if precision == "fast" else 1e-8
    max_ev = 500 if precision == "fast" else 10000

    fit_start = time.time()
    result = minimize(residual, params, nan_policy='omit',
                      args=args, method='least_squares',
                      max_nfev=max_ev, ftol=tol, gtol=tol, xtol=tol, 
                      loss='huber', diff_step=1e-4, tr_solver='lsmr', verbose=0)

    # if savepath is not None:
    #     save_result = {
    #         'country': label,
    #         'target': target,
    #         'grid_beta': params['beta_s_base'].value,
    #         'grid_sigma': params['sigma'].value,
    #         'grid_gamma': params['gamma_s'].value,
    #         'residual': result.chisqr,
    #         'opt_params': extract_param_values(result.params) # nested dict
    #         }
    #     json_fitpath = savepath.joinpath(f"seir_{filename}fit_results.json")
    #     with open(json_fitpath, "a") as f:
    #         f.write(json.dumps(save_result) + ",\n")

    return result

def simulate_SEIR(t_data, label, target, params, N, savepath=None, filename="", erlang=True):
    """
    simulate SEIR with given params 

    Arguments:
    results -- solution for given set of parameters

    Returns:
    solution of minimization 
    """
    
    # Extract optimized initial conditions
    Is0 = params['Is0'].value
    Ia0 = params['Ia0'].value
    E0  = params['E0'].value

    if erlang:
        # S, E1, E2, E3, Is, Ia, R, D (8 compartments)
        S0 = N - (E0 + Is0 + Ia0) 
        y0_opt = [S0, E0, 0, 0, Is0, Ia0, 0, 0]
        model_func = seir_erlang_model
        cols = ['S', 'E1', 'E2', 'E3', 'Is', 'Ia', 'R', 'D']
    else:
        S0 = N - (E0 + Is0 + Ia0) 
        y0_opt = [S0, E0, Is0, Ia0, 0, 0]
        model_func = seir_model
        cols = ['S', 'E', 'Is', 'Ia', 'R', 'D']

    soln = odeint(model_func, y0_opt, t_data, args=(params,))
    soln_df = pd.DataFrame(soln, columns=cols)
    soln_df['country'] = label
    soln_df['target'] = target
    soln_df['t'] = t_data 

    if savepath:
        json_bestfitpath = savepath.joinpath(f"seir_{filename}_solution.json")
        with open(json_bestfitpath, "a") as f:
            soln_df.to_json(f, orient='records', lines=True)
            f.write("\n")

    return soln_df['Is'] # predicted Symptomatic Infected (Is) from the SEIR simulation

# Define the SEIR model with symptomatic and asymptomatic cases
def seir_model(y, t, params):
    """
    SEIR model with time-dependent beta (sinusoidal variation with a 6-month period).

    Arguments:
    t -- Time step
    y -- Current values of state variables [S, E, Is, Ia, R, D]
    params -- Model parameters from lmfit

    Returns:
    List of differential equations for each compartment
    """
    # print(f"t = {t:.2f}, sum = {sum([x/60500000 for x in y]):.6f}") # sums to 1 for Italy with popn 60500000
    S, E, Is, Ia, R, D = y

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

    N = S + E + Is + Ia + R + D               # Total population

    # Time-dependent transmission rates with sinusoidal variation
    omega = 2 * np.pi / (30*4)  # Frequency corresponding to a 6-month period
    
    # Time dependent infection rate
    beta_s = beta_s_base * (1 + beta_s_amp * np.sin(omega * t))
    beta_a = beta_a_base * (1 + beta_a_amp * np.sin(omega * t))
    
    # Constante rate
    # beta_s = beta_s_base * 1
    # beta_a = beta_a_base * 1 

    # Differential equations
    dS = - (beta_s * Is + beta_a * Ia) * S / N
    dE = (beta_s * Is + beta_a * Ia) * S / N - sigma * E
    dIs = alpha * sigma * E - (gamma_s + mu) * Is
    dIa = (1 - alpha) * sigma * E - gamma_a * Ia
    dR = gamma_s * Is + gamma_a * Ia
    dD = mu * Is
    
    # print([dS, dE, dIs, dIa, dR, dD])
    return [dS, dE, dIs, dIa, dR, dD]

def seir_erlang_model(y, t, params):

    # 8 compartments instead of 6
    S, E1, E2, E3, Is, Ia, R, D = y

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
    
    # Time dependent infection rate
    beta_s = beta_s_base * (1 + beta_s_amp * np.sin(omega * t))
    beta_a = beta_a_base * (1 + beta_a_amp * np.sin(omega * t))

    # --- Erlang Logic ---
    # keep the MEAN incubation time the same (1/sigma), 
    # the rate of leaving each of the 'k' stages must be k * sigma.
    k = 3 
    transition_rate = k * sigma 

    # Differential equations
    # 1. New infections enter the first stage E1
    dS = - (beta_s * Is + beta_a * Ia) * S / N
    dE1 = (beta_s * Is + beta_a * Ia) * S / N - transition_rate * E1
    
    # 2. People flow through the 'tunnel' of E compartments
    dE2 = transition_rate * E1 - transition_rate * E2
    dE3 = transition_rate * E2 - transition_rate * E3
    
    # 3. People leave the LAST stage (E3) to become Is or Ia
    dIs = alpha * (transition_rate * E3) - (gamma_s + mu) * Is
    dIa = (1 - alpha) * (transition_rate * E3) - gamma_a * Ia
    
    dR = gamma_s * Is + gamma_a * Ia
    dD = mu * Is
    
    return [dS, dE1, dE2, dE3, dIs, dIa, dR, dD]

def g(t, y0, params, erlang=True):
    if erlang:
        # Uses the 8-compartment Erlang model
        return odeint(seir_erlang_model, y0, t, args=(params,))
    else:
        # Uses the original 6-compartment model
        return odeint(seir_model, y0, t, args=(params,))

def joint_residual(param, t, I_official, I_twitter, N):
    # 1. Run the same biological model once
    sol = g(t, y0_opt, param)
    I_true = sol[:, 4]  # The "True" infectious curve in the population

    # 2. Extract separate reporting rates for each data source
    # Official might report 5% of cases, Twitter might report 0.01%
    rho_off = param['rho_official'].value 
    rho_twit = param['rho_twitter'].value

    # 3. Calculate residuals for BOTH streams
    # We compare the single model against both datasets simultaneously
    res_official = (I_true * rho_off) - I_official
    res_twitter = (I_true * rho_twit) - I_twitter

    # 4. Concatenate them into one long array for the optimizer
    # Weighting twitter at 0.5 helps if it's noisier than official data
    return np.concatenate([res_official.ravel(), res_twitter.ravel() * 0.5])

