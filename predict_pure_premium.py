import pandas as pd
import numpy as np

def predict_pure_premium(age_band, area, freq_dict, sev_dict): 
    # take the exponentiated value of each beta for the true estimated values 
    # because the regression was log-linear
    estimated_freqs = {key: np.exp(value) for key, value in freq_dict.items()}
    estimated_sevs = {key: np.exp(value) for key, value in sev_dict.items()}
    age_key = ''
    area_key = ''
    if age_band == '18-21': 
        age_key = 'intercept' 
    else: 
        age_key = age_band

    if area == 'C': 
        area_key = 'intercept'
    else: 
        area_key = 'intercept'

    














    




    return 0