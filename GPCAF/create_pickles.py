import pandas as pd

#Create train pickles
d = pd.read_csv('../train_ads.csv')

column_names=[ 'GPCAF_Feature1', 'GPCAF_Feature2', 'GPCAF_Feature3', 'GPCAF_Feature4', 'GPCAF_Feature5', 'GPCAF_Feature6', 'GPCAF_Feature7', 'GPCAF_Feature8', 'GPCAF_Feature9', 'GPCAF_Feature10', 'GPCAF_Feature11', 'GPCAF_Feature12', 'GPCAF_Feature13', 'GPCAF_Feature14', 'GPCAF_Feature15', 'GPCAF_Feature16', 'GPCAF_Feature17', 'GPCAF_Feature18', 'GPCAF_Feature19', 'GPCAF_Feature20']
 
x_train =d[column_names]

y_train = d['reactionEnergy'] 
 
y_train = y_train.to_frame()

#Create test pickles
d1 = pd.read_csv('../test_ads.csv')

x_test =d1[column_names]

y_test = d1['reactionEnergy'] 

y_test = y_test.to_frame()

# Save test and train pickles
x_train.to_pickle('./x_train.pkl')
y_train.to_pickle('./y_train.pkl')
print('x_train and y_train saved')

x_test.to_pickle('./x_test.pkl')
y_test.to_pickle('./y_test.pkl')
print('x_test and y_test saved')
