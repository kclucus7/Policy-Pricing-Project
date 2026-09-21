import pandas as pd
import numpy as np
from weighted_regression import weighted_regression

data_path = 'rating-cell-data.csv'
frequency_beta_mapping = weighted_regression(data_path, 'frequency_raw', 'total_exposure')
severity_beta_mapping = weighted_regression(data_path, 'severity_raw', 'total_claims')






