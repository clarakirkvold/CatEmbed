import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
#from keras.layers.normalization.batch_normalization import BatchNormalization
import json
import pickle
import os
from catAdsorb import get_label_encoded_data, get_embeddings,fit_transform
import math   
d = pd.read_csv('train_val_ads.csv')

column_names = d.columns.tolist()
print(column_names)
column_names= ['density', 'band_gap', 'efermi', 'formation_energy_per_atom', 'total_magnetization', 'volume', 'energy_per_atom','facet','ads_1','sites_1', 'IE_1', 'H_1', 'S_1',
               'Equation', 'coverages','surfaceComposition'
               ]

xx =d[column_names]
yy = d['reactionEnergy'] * 23.0609 #to kcal/mol

#preprocess
def get_embedding_info(data, categorical_variables=None):
    '''
    this function identifies categorical variables and its embedding size

    :data: input data [dataframe]
    :categorical_variables: list of categorical_variables [default: None]
    if None, it automatically takes the variables with data type 'object'

    embedding size of categorical variables are determined by minimum of 50 or half of the no. of its unique values.
    i.e. embedding size of a column  = Min(50, # unique values of that column)
    '''
    if categorical_variables is None:
        categorical_variables = data.select_dtypes(include='object').columns
         
    return {col:(data[col].nunique(),min(50,(data[col].nunique()+ 1) //2)) for col in categorical_variables}

embedding_info = get_embedding_info(xx)
print(embedding_info)
X_encoded,encoders,labels = get_label_encoded_data(xx)

embeddings = get_embeddings(X_encoded, yy, categorical_embedding_info=embedding_info,
                            is_classification=False, epochs=1000)

x_train = fit_transform(X_encoded, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)

y_train = yy.to_frame()

print('Embedding complete')
with open('./embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings, file)
    
with open('./encoders.pkl', 'wb') as file:
    pickle.dump(encoders, file)
    
with open('./labels.pkl', 'wb') as file:
    pickle.dump(labels, file)
    
x_train.to_pickle('./x_train.pkl')

y_train.to_pickle('./y_train.pkl')

print('Saved x_train and y_train')
print('Starting training')

from catboost import CatBoostRegressor
import joblib
model = CatBoostRegressor(iterations=15000, learning_rate=0.1, objective='RMSE', depth=8, bootstrap_type='Bernoulli', subsample=1.0, sampling_frequency='PerTree', langevin=True, diffusion_temperature=20000, leaf_estimation_iterations=2, leaf_estimation_backtracking='AnyImprovement')
_ = model.fit(x_train, y_train)

#save model
joblib.dump(model, './model.pkl')

#Get Test
test = pd.read_csv('train_val_ads.csv val_ads.csv')
x_test =test[column_names]
y_test = test['reactionEnergy'] * 23.0609 #to kcal/mol
y_test = y_test.to_frame()
# Get the names of columns to be transformed
categorical_variables = [col for col in xx.columns if xx[col].dtype == 'object']
#Transform Dataframe into encoded data
for col in categorical_variables:
    x_test[col] = xx[col].apply(lambda x: label_mappings[col][x] if x in label_mappings[col] else x)

#Transform Dataframe into embedding data
x_test = ce.fit_transform(x_test, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)

#Save transfromed embedding data
x_test.to_pickle('./externalx.pkl')
y_test.to_pickle('./y_test.pkl')

# Make predictions
y_pred = model.predict(x_test)

# R2
print(r2_score(y_test, y_pred), 'r2 test')

# Range of values in y_test
range_test = float(np.max(y_test) - np.min(y_test))

# RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)), "test RMSE")

# MAE
print(mean_absolute_error(y_test, y_pred), "test MAE")

# MDAE
print(median_absolute_error(y_test, y_pred), "test MDAE")

# MAX_ERROR
print(max_error(y_test, y_pred), "test MAX_ERROR")

# Create DataFrame with y_pred, y_test, and ads_1
result_df = pd.DataFrame({
    'y_pred': y_pred.squeeze(),
    'y_test': y_test.squeeze(),
    'ads_1': test['ads_1'],
    'sites_1': test['sites_1'],
    'surfaceComposition': test['surfaceComposition']
})

# Save DataFrame to a CSV file
current_directory = os.getcwd()
directory_name = os.path.basename(current_directory)
result_df.to_csv(f'../csvs/{directory_name}-predictions.csv', index=False)

