#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import categorical_entity_embedder as ce
import pickle

d = pd.read_csv('../test_react.csv')

column_names = d.columns.tolist()

column_names= ['Equation','surfaceComposition']

xx =d[column_names]
yy = d['reactionEnergy'] 

y_test = yy.to_frame()

#Load embeddings, encoders, and label mappings obtained from data_transform.py

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



