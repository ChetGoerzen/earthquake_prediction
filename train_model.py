import xgboost as xgb
import pickle

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + "20161105_df.p", "rb") as f:
    X_train = pickle.load(f)

with open(data_loc + "y.p", "rb") as f:
    Y_train = pickle.load(f)

D_train = xgb.DMatrix(X_train, label=Y_train)

param = {
    "eta": 0.1,
    "max_depth": 8,
    "objective": "reg:squarederror"
}

steps = 500

model = xgb.train(param, D_train, steps)

pickle.dump(model, open(data_loc + "model.p", "wb"))
