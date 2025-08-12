#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:53:22 2025

@author: llopez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer
import pandas as pd
# 🔹 Define the SEIR model with symptomatic and asymptomatic cases
def seir_model(t, y, params):
    """
    SEIR model with symptomatic, asymptomatic, recovered, and deceased individuals.

    Arguments:
    t -- Time step
    y -- Current values of state variables [S, E, Is, Ia, R, D]
    params -- Model parameters from lmfit

    Returns:
    List of differential equations for each compartment
    """
    S, E, Is, Ia, R, D = y

    # Extract parameters from lmfit
    beta_s = params['beta_s'].value  # Transmission rate (symptomatic)
    beta_a = params['beta_a'].value  # Transmission rate (asymptomatic)
    sigma = params['sigma'].value    # Incubation rate (E -> I)
    gamma_s = params['gamma_s'].value # Recovery rate (symptomatic)
    gamma_a = params['gamma_a'].value # Recovery rate (asymptomatic)
    mu = params['mu'].value           # Mortality rate (symptomatic)
    alpha = params['alpha'].value      # Proportion of symptomatic infections

    N = S + E + Is + Ia + R + D  # Total population

    # Differential equations
    dS = - (beta_s * Is + beta_a * Ia) * S / N
    dE = (beta_s * Is + beta_a * Ia) * S / N - sigma * E
    dIs = alpha * sigma * E - (gamma_s + mu) * Is
    dIa = (1 - alpha) * sigma * E - gamma_a * Ia
    dR = gamma_s * Is + gamma_a * Ia
    dD = mu * Is

    return [dS, dE, dIs, dIa, dR, dD]

# ================================================================================

def objective_function(params, t_data, I_real, C_real):
    """
    Objective function for fitting the SEIR model to real influenza data.
    
    Arguments:
    params -- lmfit Parameters object containing model parameters to optimize
    t_data -- Time points corresponding to real data
    I_real -- Observed reported cases at each time step
    C_real -- Observed cumulative cases at each time step

    Returns:
    Residuals (errors) combining absolute differences in reported and cumulative cases,
    as well as peak timing misalignment.
    """
    # Solve the SEIR model with current parameter values
    y0 = [S0, E0, Is0, Ia0, R0, D0]  # Initial conditions
    sol = solve_ivp(seir_model, (t_data[0], t_data[-1]), y0, t_eval=t_data, args=(params,))
    
    # Extract model-predicted values
    I_model = sol.y[2]  # Symptomatic Infected (Is)
    C_model = np.cumsum(I_model)  # Cumulative cases over time

    # Calculate Peak Infection Time
    peak_model_t = t_data[np.argmax(I_model)]  # Time at which peak infection occurs
    peak_real_t = t_data[np.argmax(I_real)]    # Time at which real peak occurs
    
    # Compute residuals for:
    # 1. Case count error (RMSE)
    case_residual = (I_model - I_real) / np.max(I_real)  # Normalized error

    # 2. Cumulative case error (ensuring the model fits total outbreak dynamics)
    cumulative_residual = (C_model - C_real) / np.max(C_real)  # Normalized error

    # 3. Peak timing error (ensuring the model captures outbreak timing)
    # peak_residual = (peak_model_t - peak_real_t) / peak_real_t  # Relative peak time shift

    # Combine residuals into a single loss function
    # loss = np.concatenate([case_residual, cumulative_residual, [peak_residual]])
    loss = np.concatenate([case_residual, cumulative_residual])


    return loss
# ================================================================================

# 🔹 Define model parameters using lmfit
params = Parameters()
params.add('beta_s', value=np.random.uniform(0.1, 0.5), min=0.1, max=5)  # Transmission rate (symptomatic)
params.add('beta_a', value=np.random.uniform(0.05, 1.5), min=0.05, max=1.5)  # Transmission rate (asymptomatic)
params.add('sigma', value=np.random.uniform(1/5, 1/2), min=1/5, max=1/2)   # Incubation rate (E → I)
params.add('gamma_s', value=np.random.uniform(1/14, 1/3), min=1/14, max=1/3) # Recovery rate (symptomatic)
params.add('gamma_a', value=np.random.uniform(1/10, 1/3), min=1/10, max=1/3) # Recovery rate (asymptomatic)
params.add('mu', value=np.random.uniform(0, 0.1), min=0, max=0.1)   # Mortality rate (symptomatic)
params.add('alpha', value=np.random.uniform(0.4, 0.9), min=0.4, max=0.9)   # Proportion of symptomatic infections


file_path = "/home/llopez/Descargas/FluData/weekly-confirmed-cases-of-influenza Spain.csv"
df_influenza = pd.read_csv(file_path)


df_fit = df_influenza[(df_influenza['Country'] == 'Spain') & (df_influenza['Year'] == 2018)]

df_fit['t']=(np.arange(0, len(df_fit)))*7
df_fit=df_fit[df_fit['t']<=24*7]
# I'm takinf just the first 24 weeks of data.
# Load the CSV file

I_real=df_fit['All strains - All types of surveillance'].values
C_real=I_real.cumsum()
t_data = df_fit['t'].values

# 🔹 Initial conditions (Population = 1,000,000)
N = 46e6
I0 = I_real[0]    # Initial infections
E0 = I0*2    # Initial exposed individuals
S0 = N - I0 - E0  # Remaining are susceptible
R0 = 0
D0 = 0
Ia0 = 0.5 * I0  # Half of infections are asymptomatic
Is0 = 0.5 * I0  # Half of infections are symptomatic

y0 = [S0, E0, Is0, Ia0, R0, D0]

# 🔹 Solve the system over 180 days
t_span = (min(t_data), max(t_data))  # Simulate for 180 days
t_eval = np.linspace(*t_span, max(t_data))


# Optimize parameters
# You can set up this part of the code to make a loop of N experiments with 
# initial random conditions and save the full set of parameters
result = minimize(objective_function, params, args=(t_data, I_real, C_real), method='least_squares',max_nfev=1000, nan_policy='omit',
                ftol=1e-8, gtol=1e-8, xtol=1e-8, loss='linear', diff_step=1e-4, verbose=2, tr_solver='lsmr')

# Display optimization results
# print(result.fit_report())


params=result.params
sol = solve_ivp(seir_model, t_span, y0, t_eval=t_eval, args=(params,))



# Extract results
S, E, Is, Ia, R, D = sol.y


# Ensure time points match (assuming `t_data` is the timeline used in the model)
t_data = df_fit["t"].values  # Real-world time points (days)
I_real = df_fit["All strains - All types of surveillance"].values  # Real case data

# Model predictions from the SEIR simulation
I_model = sol.y[2]  # Extract Symptomatic Infected (Is) from the model

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

# # ==========================================================================
# # 🔹 Plot Full Dynamics
# fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# # Plot Infected and Exposed individuals
# ax[0].plot(t_eval, Is, label="Symptomatic Infected (Is)", color="red")
# ax[0].plot(t_eval, Ia, label="Asymptomatic Infected (Ia)", color="purple")
# ax[0].plot(t_eval, E, label="Exposed (E)", color="orange")
# ax[0].set_title("SEIR Model - Infected and Exposed Individuals")
# ax[0].set_xlabel("Days")
# ax[0].set_ylabel("Population")
# ax[0].legend()
# ax[0].grid()

# #  Plot Susceptible, Recovered, and Deceased
# ax[1].plot(t_eval, S, label="Susceptible (S)", color="blue")
# ax[1].plot(t_eval, R, label="Recovered (R)", color="green")
# ax[1].plot(t_eval, D, label="Deceased (D)", color="black", linestyle="dashed")
# ax[1].set_title("SEIR Model - Susceptible, Recovered, and Deceased")
# ax[1].set_xlabel("Days")
# ax[1].set_ylabel("Population")
# ax[1].legend()
# ax[1].grid()

# plt.tight_layout()
# plt.show()
# # ==========================================================================