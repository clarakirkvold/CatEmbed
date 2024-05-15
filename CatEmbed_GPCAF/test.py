import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, max_error
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Label information
ads_label = pd.read_csv('../test_ads.csv')

# Read in data
x_test = pd.read_pickle('./externalx.pkl')
y_test = pd.read_pickle('./y_test.pkl')
model = joblib.load('./model.pkl')

print(x_test)

# Make predictions
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred)


# Create DataFrame with y_pred, y_test, and ads_1
result_df = pd.DataFrame({
    'y_pred': y_pred.squeeze(),
    'y_test': y_test.squeeze(),
    'adsorbate': ads_label['adsorbate'],
    'site': ads_label['site'],
    'surfaceComposition': ads_label['surfaceComposition']
})

# Save DataFrame to a CSV file
current_directory = os.getcwd()
directory_name = os.path.basename(current_directory)
result_df.to_csv(f'./{directory_name}-predictions.csv', index=False)

# Evaluate the model

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

# Plotting SI Figure 1
diff = y_test.values - y_pred.values
diff = pd.DataFrame(diff)
print("Prediction Error Summary:")
print(diff.describe())

plt.hist(diff.dropna().values, bins=40)
plt.title('Histogram of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

# Add 'ads_1' column to x_test and y_test
x_test['adsorbate'] = ads_label['adsorbate']
y_test['adsorbate'] = ads_label['adsorbate']

# Get unique adsorbates
ad_list = y_test['adsorbate'].unique()

# Evaluation loop

#os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
for ads in ad_list:
    x_split = x_test[x_test.adsorbate == ads].drop(columns=['adsorbate'])
    y_split = y_test[x_test.adsorbate == ads].drop(columns=['adsorbate'])
    
    # Ensure that data for this adsorbate exists in the test set
    if not x_split.empty and not y_split.empty:
        y_pred_s = model.predict(x_split)
        y_pred_s = pd.DataFrame(y_pred_s)
        
        print(np.sqrt(mean_squared_error(y_split, y_pred_s)), "test RMSE", ads)
        print((mean_absolute_error(y_split, y_pred_s)), "test MAE", ads)
        print((median_absolute_error(y_split, y_pred_s)), "test MDAE", ads)
        print((max_error(y_split, y_pred_s)), "test MAX_ERROR", ads)
    else:
        print(f"No data for adsorbate {ads} in the test set.")
