#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:53:22 2025

@author: llopez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer
import pandas as pd
# Define the SEIR model with symptomatic and asymptomatic cases

def seir_model(y,t, params):
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

    N = S + E + Is + Ia + R + D  # Total population

    # Time-dependent transmission rates with sinusoidal variation
    omega = 2 * np.pi / (30*4)  # Frequency corresponding to a 6-month period
    # Time dependent infection rate
    beta_s = beta_s_base * (1 + beta_s_amp * np.sin(omega * t))
    beta_a = beta_a_base * (1 + beta_a_amp * np.sin(omega * t))
    #Constante rate
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
# ================================================================================

def g(t,y0, params):
    return odeint(seir_model,y0,t, args=(params,))


# ================================================================================

# def residual(param, t, y0, data):
#     y_model = g(t,y0, param)
#     I_model = y_model[:, 2]  # Infectados sintomáticos
#     C_model = np.cumsum(I_model)  # Casos acumulados reportados

#     epsilon = 1e-10
#     max_cases = np.max(data[:])
#     min_cases = np.min(data[:])
#     denominator_cases = max_cases - min_cases
#     wmae_cases = np.mean(np.abs(I_model - data[0]) / denominator_cases) if denominator_cases > epsilon else 0

#     max_accumulated = np.max(np.cumsum(data[:]))
#     min_accumulated = np.min(np.cumsum(data[:]))
#     denominator_accumulated = max_accumulated - min_accumulated
#     wmae_accumulated = np.mean(np.abs(C_model - np.cumsum(data[:])) / denominator_accumulated) if denominator_accumulated > epsilon else 0
# # 
#     # f_real, Pxx_real = periodogram(data[:])
#     # f_model, Pxx_model = periodogram(I_model)
#     # periodogram_difference = np.mean(np.abs(Pxx_model - Pxx_real))

#     alpha = 0.8
#     beta = 0.2
#     final_residual = alpha * wmae_cases + beta * wmae_accumulated
#     final_residual = np.nan_to_num(final_residual, nan=1e6, posinf=1e6, neginf=1e6)
#     return final_residual
# ================================================================================


def residual(param, t, y0, data):
    """
    Objective function for fitting the SEIR model to real data.
    """
    y0 = [S0, E0, Is0, Ia0, R0, D0]  # Initial conditions
    sol =  g(t,y0, param)
    
    I_model = sol[:, 2]  # Symptomatic Infected (Is)
    C_model = np.cumsum(I_model)  # Cumulative reported cases
    
    Inf = ((I_model - I_real) / np.max(I_real)).ravel()
    Cum = ((C_model - C_real) / np.max(C_real)).ravel()
    
    resid = np.array([Inf * 5, Cum * 3])
    return resid

# ================================================================================

# Define parameter values
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

file_path = "/home/llopez/Descargas/FluData/weekly-confirmed-cases-of-influenza Spain.csv"
df_influenza = pd.read_csv(file_path)


df_fit = df_influenza[(df_influenza['Country'] == 'Spain') & (df_influenza['Year'] == 2019)]

df_fit['t']=(np.arange(0, len(df_fit)))*7
df_fit=df_fit[df_fit['t']<=24*6]
# I'm takinf just the first 24 weeks of data.
# Load the CSV file

I_real=df_fit['All strains - All types of surveillance'].values
C_real=I_real.cumsum()
t_data = df_fit['t'].values

# Initial conditions (Population = 1,000,000)
N = 46e6
I0 = I_real[0]# Initial infections
E0 = I0 *1    # Initial exposed individuals
S0 = N - I0 - E0  # Remaining are susceptible
R0 = 0
D0 = 0
Ia0 = I_real[0]*2 # Half of infections are asymptomatic
Is0 = I0  # Half of infections are symptomatic

y0 = [int(S0), int(E0), int(Is0), int(Ia0), int(R0), int(D0)]

# 🔹 Solve the system over 180 days
t_span = (min(t_data), max(t_data))  # Simulate for 180 days
t_eval = np.linspace(*t_span, max(t_data))


# Optimize parameters
# You can set up this part of the code to make a loop of N experiments with 
# initial random conditions and save the full set of parameters
# result = minimize(objective_function, params, args=(t_data, I_real, C_real), method='least_squares',max_nfev=10000, nan_policy='omit',
                # ftol=1e-12, gtol=1e-12, xtol=1e-12, loss='soft_l1', diff_step=1e-5, verbose=2, tr_solver='lsmr')


result = minimize(residual, params, args=(t_data,y0,I_real), method='least_squares', max_nfev=10000, nan_policy='omit',
                  ftol=1e-8, gtol=1e-8, xtol=1e-8, loss='huber', diff_step=1e-4, verbose=2, tr_solver='lsmr')
# result = minimize(residual, params, args=(t_data, y0, I_real), method='least_squares', max_nfev=100000,
                 # ftol=1e-10, gtol=1e-10, xtol=1e-10, loss='arctan', diff_step=1e-4, verbose=2, tr_solver='lsmr')

  
# Display optimization results
# print(result.fit_report())

            # param=sol.param
            # sol = minimize(
            #     residual, 
            #     param, 
            #     args=(topt_train, y0opt_train, DataM_train), 
            #     method='least_squares', 
            #     max_nfev=10000, 
            #     nan_policy='omit',
            #     ftol=1e-10, 
            #     gtol=1e-10, 
            #     xtol=1e-10, 
            #     loss='soft_l1',  # Cambiar a una función de pérdida más robusta
            #     diff_step=1e-4, 
            #     verbose=2
            # )
            
params=result.params
# sol = solve_ivp(seir_model, t_span, y0, t_eval=t_eval, args=(params,))
sol = odeint(seir_model,y0,t_data, args=(params,))


# Extract results
S, E, Is, Ia, R, D = sol[:,0],sol[:,1],sol[:,2],sol[:,3],sol[:,4],sol[:,5]


# Ensure time points match (assuming `t_data` is the timeline used in the model)
t_data = df_fit["t"].values  # Real-world time points (days)
I_real = df_fit["All strains - All types of surveillance"].values  # Real case data

# Model predictions from the SEIR simulation
I_model = Is  # Extract Symptomatic Infected (Is) from the model

# Plot Real Data vs. Model Predictions
plt.figure(figsize=(10, 6))

plt.plot(t_data, I_real, 'bo-', label="Real Data (Influenza Cases)", markersize=4)
plt.plot(t_data, I_model[:len(t_data)], 'r--', label="Model Prediction (SEIR)", linewidth=2)

plt.xlabel("Days")
plt.ylabel("Number of Cases")
plt.title("Real Influenza Cases vs. SEIR Model Prediction")
plt.legend()
plt.grid()

plt.show()

