
import random
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../ads_data.csv')

for i in range(5):
 filtered_df = df[df['RatioB'] != 0]
 pure_metals= df[df['RatioB'] == 0]
 # Calculate the desired number of samples for the test set
 num_samples_test = int(len(df) * 0.2)

 # Initialize variables
 test_data = pd.DataFrame()
 remaining_data = filtered_df.copy()

 # Randomly select surface compositions for the test set until the desired number of samples is reached
 while len(test_data) < num_samples_test:

    # Get unique surface compositions
    unique_surface_compositions = filtered_df['surfaceComposition'].unique()

    # Randomly select a surface composition
    selected_composition = random.choice(unique_surface_compositions)

    # Add all rows with the selected composition to the test set
    test_data = pd.concat([test_data, remaining_data[remaining_data['surfaceComposition'] == selected_composition]])

    # Remove the selected composition from the list of unique compositions
    unique_surface_compositions = unique_surface_compositions[unique_surface_compositions != selected_composition]

    # Remove selected rows from the remaining data
    remaining_data = remaining_data[remaining_data['surfaceComposition'] != selected_composition]

 # The remaining data is for training
 train_data = pd.concat([remaining_data, pure_metals], ignore_index=True)

 # Check if there are any common surface compositions between train and test sets
 common_compositions = set(train_data['surfaceComposition']).intersection(set(test_data['surfaceComposition']))

 if common_compositions:
    print("Error: Train and test sets have common surface compositions.")
    # Optionally, you can handle this situation by re-running the random selection process or by taking other corrective measures.
 else:
    print("Train and test sets have no common surface compositions. Proceeding with further processing.")
    # Proceed with further processing, such as saving the train and test sets into CSV files.


 #folder='embedded_reaction_energies'
 folder=i+1
 suf='sur'
 print(folder)
 train_df=train_data
 test_df=test_data 
 print(train_df)
 print(test_df)
 # Save the train and test DataFrames to CSV files in respective folders
 train_df.to_csv(f'embedded/{folder}/train_{suf}.csv', index=False)
 test_df.to_csv(f'embedded/{folder}/test_{suf}.csv', index=False)

 train_df.to_csv(f'embedded_GPCAF/{folder}/train_{suf}.csv', index=False)
 test_df.to_csv(f'embedded_GPCAF/{folder}/test_{suf}.csv', index=False)


 train_df.to_csv(f'GPCAF/{folder}/train_{suf}.csv', index=False)
 test_df.to_csv(f'GPCAF/{folder}/test_{suf}.csv', index=False)

 train_df.to_csv(f'GPCAF_ads/{folder}/train_{suf}.csv', index=False)
 test_df.to_csv(f'GPCAF_ads/{folder}/test_{suf}.csv', index=False)

