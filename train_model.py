import xgboost as xgb
import pickle
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + "aggregate_X.p", "rb") as f:
    X = pd.DataFrame(pickle.load(f))
X.reset_index(drop=True, inplace=True)
#print(len(X))

with open(data_loc + "y_large.p", "rb") as f:
    y = pd.Series(pickle.load(f))
y.reset_index(drop=True, inplace=True)
#print(len(y))

y = y[0:len(X)]
#print(len(X))
#print(len(y))
print(X)
print(y)
stop = int(len(X) * 0.85)

X_train = X[0:stop]
y_train = y[0:stop]

#print(len(X_train))

X_test = X[stop:-1]
y_test = y[stop:-1]

#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)

print(f"Test/Train Ratio: {len(X_test) / len(X_train)}")

#model = xgb.train(param, D_train, steps)
model = xgb.XGBRegressor(eta=0.1, max_depth=8, n_estimators=500)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="rmse",
          eval_set=eval_set, verbose=True)

pickle.dump(model, open(data_loc + "model.p", "wb"))

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mae(train_pred, y_train)
test_mae = mae(test_pred, y_test)

print(f"Training MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print()

train_corr = pearsonr(train_pred, y_train)
test_corr = pearsonr(test_pred, y_test)

print(f"Train Prediction Correlation: {train_corr}")
print(f"Test Prediction Correlation: {test_corr}")
