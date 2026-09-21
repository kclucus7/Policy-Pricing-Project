import pandas as pd
import numpy as np


def weighted_regression(data_path, outcome, weight): 
    df = pd.read_csv(data_path)
    # reference levels are age_band of 41-50 and area_density_type of C as this is 
    # the most represented bucket in the data, so listed first and drop_first 
    # excludes from dummies
    df['age_band'] = pd.Categorical(df['age_band'], categories = ['41-50', '18-21', '22-30', '31-40', '51-60', '61-70', '71-80', '81-100'])
    df['area_density_type'] = pd.Categorical(df['area_density_type'], categories = ['C', 'A', 'B', 'D', 'E', 'F'])

    # create dummy variable for all combinations of age band and area density type 
    # (excluding our reference which will appear as false, false)
    dummies = pd.get_dummies(df, columns = ['age_band', 'area_density_type'], drop_first= True)

    # build regression elements for log regression

    # regressing to find predicted severity given bucketing system
    Y = np.log(df[outcome])

    # filter age band dummy columns
    age_band_dummy_columns = dummies.filter(like = 'age_band_')

    # filter area density type dummy columns
    area_density_type_dummy_columns = dummies.filter(like = 'area_density_type_')

    # concetenate dummy columns to make the matrix with dummy inputs representing 
    # combinations of age brackets and area density types
    dummy_columns = pd.concat([age_band_dummy_columns, area_density_type_dummy_columns], axis = 1)

    # create dummy column representing intercepts (coefficients of B0) as 1 
    # everywhere with same length as the dummy columns 
    intercept_column = pd.Series(1, index =range(len(dummy_columns)), name = 'intercept')

    # concetenate the intercept column with dummy columns to aggregate x
    X = pd.concat([intercept_column, dummy_columns], axis=1)

    # capture the column names for labelling purposes later
    x_columns = X.columns

    # convert x to numpy array so it's compatible for matrix operations
    X = X.to_numpy().astype(dtype = 'float64')


    # build weights matrix, a diagonal matrix with exposures as weights
    weights = pd.Series(df[weight]).to_numpy()
    W = np.diag(weights)

    # build beta using precalcualted formula
    severity_beta = np.linalg.inv(X.T@W@X)@X.T@W@Y

    # map beta values back to column names to make model outcomes more paletable
    beta_mapping = dict(zip(x_columns, severity_beta))
    return beta_mapping