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

from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Output to stdout
    )
logger = logging.getLogger()

def extract_param_values(params):
    """Extract just the .value of each lmfit Parameter into a flat dict."""
    return {k: v.value for k, v in params.items()}

# fit seir_model to data
def fit_seir(df, target, N, label, savepath):
    
    # data values
    I_real = df[target].values # real incidences
    t_data = df['t'].values    # time stamp
    T = df.shape[0]

    # Initial conditions 
    # N   = 46e6       # (Population = 1,000,000)
    I0  = I_real[0]    # Initial infections
    E0  = I0*1         # Initial exposed individuals
    S0  = N - I0 - E0  # Remaining are susceptible
    R0  = 0
    D0  = 0            # ?? what is D
    Ia0 = I_real[0]*2  # Half of infections are asymptomatic
    Is0 = I0           # Half of infections are symptomatic
    y0 = [int(S0), int(E0), int(Is0), int(Ia0), int(R0), int(D0)]

    # Solve the system over T*7 days
    t_span = (min(t_data), max(t_data))          
    t_eval = np.linspace(*t_span, max(t_data))   

    # Optimize parameters
    # TO DO - move to config
    beta_s_base = np.linspace(0.2, 1.5, 5)
    sigma = np.linspace(0.2, 0.5, 4)
    gamma_s = np.linspace(0.1, 0.5, 5)
    param_grid = list(product(beta_s_base, sigma, gamma_s))
    param_grid = random.sample(param_grid, 2)
    print(f"Number of fits: {len(param_grid)}")
    
    results = []
    start_time = time.time()
    try:
        for i, (beta, sigma, gamma) in enumerate(tqdm(param_grid, desc="Fitting models")):
            # Define model parameters using lmfit
            # TO DO - move to config
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
            
            # optimize
            fit_start = time.time()
            result = minimize(residual, params, nan_policy='omit',
                        args=(t_data, y0, I_real), method='least_squares',
                        max_nfev=10000, ftol=1e-8, gtol=1e-8, xtol=1e-8, loss='huber',
                        # max_nfev=100000, ftol=1e-10, gtol=1e-10, xtol=1e-10, loss='arctan', 
                        diff_step=1e-4, tr_solver='lsmr', 
                        verbose=0)
            results.append(result)
            fit_duration = time.time() - fit_start
            if (i + 1) % 10 == 0 or (i + 1) == len(param_grid):
                tqdm.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Fit {i + 1}: "
                    f"beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, "
                    f"residual={result.chisqr:.4f}, duration={fit_duration:.2f}s"
                )
            # logger.info(f"Fit {i + 1}: beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, residual={result.chisqr:.4f}")

            # save result
            save_result = {
                'country': label,
                'target': target,
                'beta': beta,
                'sigma': sigma,
                'gamma': gamma,
                'residual': result.chisqr,
                'opt_params': extract_param_values(result.params) # nested dict
                }
            json_fitpath = savepath.joinpath("seir_fit_results.json")
            with open(json_fitpath, "a") as f:
                f.write(json.dumps(save_result) + ",\n")

    except Exception as e:
        logger.warning(f"Fit {i + 1} failed: {e}")
    total_duration = time.time() - start_time
    logger.info(f"All fits completed in {total_duration:.2f} seconds")
    
    if results:
        # obtain and save results with best params
        best_result = min(results, key=lambda r: r.chisqr)
        best_params = best_result.params  # <--- this is the Parameters object
        logger.info(
            f"Best fit: beta={best_params['beta_s_base'].value:.3f}, "
            f"sigma={best_params['sigma'].value:.3f}, "
            f"gamma={best_params['gamma_s'].value:.3f}, "
            f"residual={best_result.chisqr:.4f}"
        )

        # Simulate SEIR model
        soln = odeint(seir_model, y0, t_data, args=(best_params,))
        soln_df = pd.DataFrame(soln, columns=['S', 'E', 'Is', 'Ia', 'R', 'D'])
        soln_df['country'] = label
        soln_df['target'] = target
        
        json_bestfitpath = savepath.joinpath("seir_best_solution.json")
        with open(json_bestfitpath, "a") as f:
            soln_df.to_json(f, orient='records', lines=True)
            f.write("\n")

        I_model = soln_df['Is'] # predicted Symptomatic Infected (Is) from the SEIR simulation
        I_pred = I_model[:T]
        return I_pred
    else:
        logger.error("No successful fits found.")
        return None

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

def g(t, y0, params):
    return odeint(seir_model, y0, t, args=(params,))

def residual(param, t, y0, I_real):
    """
    Objective function for fitting the SEIR model to real data.
    """
    # y0 = [S0, E0, Is0, Ia0, R0, D0]  # Initial conditions
    C_real = I_real.cumsum()                               # cumulative incidences

    sol =  g(t, y0, param)
    # sol = odeint(seir_model, y0, t_data, args=(params,))
    
    I_model = sol[:, 2]  # Symptomatic Infected (Is)
    C_model = np.cumsum(I_model)  # Cumulative reported cases
    
    Inf = ((I_model - I_real) / np.max(I_real)).ravel()
    Cum = ((C_model - C_real) / np.max(C_real)).ravel()
    
    resid = np.array([Inf * 5, Cum * 3])
    return resid

