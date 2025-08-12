#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:53:22 2025

@author: llopez
"""

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer

# fit seir_model to data
def fit_seir(df, target, params):
    I_real = df[target].values # real incidences
    t_data = df['t'].values    # time stamp

    # Initial conditions (Population = 1,000,000)
    N   = 46e6
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
    # You can set up this part of the code to make a loop of N experiments with 
    # initial random conditions and save the full set of parameters
    result = minimize(residual, params, nan_policy='omit',
                      args=(t_data, y0, I_real), method='least_squares',
                      max_nfev=10000, ftol=1e-8, gtol=1e-8, xtol=1e-8, loss='huber',
                      # max_nfev=100000, ftol=1e-10, gtol=1e-10, xtol=1e-10, loss='arctan', 
                      diff_step=1e-4, tr_solver='lsmr', 
                      verbose=2)
    params = result.params

    # sol = solve_ivp(seir_model, t_span, y0, t_eval=t_eval, args=(params,))
    sol = odeint(seir_model,y0,t_data, args=(params,))
    return sol

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

