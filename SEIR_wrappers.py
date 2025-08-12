#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 7 2025

@author: nitimkc
"""

import numpy as np
import pandas as pd
from lmfit import minimize
from ILI_SEIR import residual

def getseason(week, year, N=40):
    # influenza season
    if week>=N:
        season = f"{year}{str(year+1)[-2:]}" # beginning of the season
    else:               
        season = f"{year-1}{str(year)[-2:]}" # end of the season
        
    return season
# getseason(5,2009)

