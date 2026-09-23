import pandas as pd
import numpy as np

def predict_pure_premium(age_band, area, freq_dict, sev_dict): 
    # take the exponentiated value of each beta for the true estimated values 
    # because the regression was log-linear
    estimated_freqs = {key: np.exp(value) for key, value in freq_dict.items()}
    estimated_sevs = {key: np.exp(value) for key, value in sev_dict.items()}
    age_key = 'age_band_' + age_band
    area_key = 'area_density_type_' + area
    predicted_freq = estimated_freqs['intercept']*estimated_freqs.get(age_key, 1.0)*estimated_freqs.get(area_key, 1.0)
    predicted_sev = estimated_sevs['intercept']*estimated_sevs.get(age_key, 1.0)*estimated_sevs.get(area_key, 1.0)
    return predicted_freq*predicted_sev














    




    return 0