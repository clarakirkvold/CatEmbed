import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor

x_train=pd.read_pickle('./x_train.pkl')
y_train=pd.read_pickle('./y_train.pkl')
print(x_train)

model = CatBoostRegressor(iterations=15000, learning_rate=0.1, objective='RMSE', depth=8, bootstrap_type='Bernoulli', subsample=1.0, sampling_frequency='PerTree', langevin=True, diffusion_temperature=20000, leaf_estimation_iterations=2, leaf_estimation_backtracking='AnyImprovement')
_ = model.fit(x_train, y_train)

#save model
joblib.dump(model, './model.pkl')
