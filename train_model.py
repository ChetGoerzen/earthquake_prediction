import xgboost as xgb
import pickle

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + "df.p", "rb") as f:
    X_train = pickle.load(f)

with open(data_loc + "y.p", "rb") as f:
    Y_train = pickle.load(f)

D_train = xgb.DMatrix(X_train, label=Y_train)

param = {
    "eta": 0.3,
    "max_depth": 10,
    "objective": "reg:squarederror"
}

steps = 1000

model = xgb.train(param, D_train, steps)

pickle.dump(model, open(data_loc + "model.p", "wb"))
