#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:41:41 2023

@author: clarakirkvold
"""
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
import categorical_embedder as ce
import json
import pickle

d = pd.read_csv('test_ads.csv')

column_names = d.columns.tolist()
print(column_names)
column_names= ['density', 'band_gap', 'efermi', 'formation_energy_per_atom', 'total_magnetization', 'volume', 'energy_per_atom','facet','ads_1','sites_1', 'IE_1', 'H_1', 'S_1',
               'Equation', 'coverages','surfaceComposition'
               ]
xx =d[column_names]
yy = d['reactionEnergy'] * 23.0609 #to kcal/mol

y_test = yy.to_frame()

#Load embeddings, encoders, and label mapping obtained from data_transform.py

with open('./embeddings.pkl', 'rb') as file:
    embeddings = pickle.load(file)
    
with open('./encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

with open('./labels.pkl', 'rb') as file:
    label_mappings = pickle.load(file)


# Get the names of columns to be transformed
categorical_variables = [col for col in xx.columns if xx[col].dtype == 'object']

#Transform Dataframe into encoded data
for col in categorical_variables:
    xx[col] = xx[col].apply(lambda x: label_mappings[col][x] if x in label_mappings[col] else x)

#Transform Dataframe into embedding data
x_test = ce.fit_transform(xx, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)

#Save transfromed embedding data
x_test.to_pickle('./externalx.pkl')

y_test.to_pickle('./y_test.pkl')
print('Saved transfromed embedding data')



