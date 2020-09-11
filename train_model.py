import xgboost as xgb
import pickle

with open("./data/df.p", "rb") as f:
    X_train = pickle.load(f)

with open("./data/y.p", "rb") as f:
    Y_train = pickle.load(f)

D_train = xgb.DMatrix(X_train, label=Y_train)

param = {
    "eta": 0.3,
    "max_depth": 100,
    "objective": "reg:squarederror"
}

steps = 2000

model = xgb.train(param, D_train, steps)

pickle.dump(model, open("./data/model.p", "wb"))
