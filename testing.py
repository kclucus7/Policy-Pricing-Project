import pandas as pd
import numpy as np
from weighted_regression import weighted_regression
from predict_pure_premium import predict_pure_premium
import itertools

data_path = 'rating-cell-data.csv'
frequency_beta_mapping = weighted_regression(data_path, 'frequency_raw', 'total_exposure')
severity_beta_mapping = weighted_regression(data_path, 'severity_raw', 'total_claims')
age_bands = ['18-21', '22-30', '31-40','41-50', '51-60', '61-70', '71-80', '81-100']
area_types = ['A', 'B', 'C', 'D', 'E', 'F']
buckets = list(itertools.product(age_bands, area_types))
predicted_premiums = {}
for bucket in buckets: 
    predicted_premiums[bucket] = predict_pure_premium(bucket[0], bucket[1], frequency_beta_mapping, severity_beta_mapping)

print(predicted_premiums)




