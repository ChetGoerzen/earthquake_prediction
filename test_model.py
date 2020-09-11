import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open("./model.p", "rb") as f:
    model = pickle.load(f)

with open(data_loc + "df.p", "rb") as f:
    X_train = pickle.load(f)

with open(data_loc + "y.p", "rb") as f:
    Y_train = pickle.load(f)

D_train = xgb.DMatrix(X_train, label=Y_train)

preds = model.predict(D_train)

fig, ax = plt.subplots(dpi=500)
ax.plot(preds, "r")
ax.plot(Y_train, "b")
plt.xlim(0, 500)
plt.savefig("./images/fit_to_train.png", bbox_inches='tight', dpi=500)
plt.show()

fig, ax = plt.subplots()
xgb.plot_importance(model, ax=ax, max_num_features=25)
plt.savefig("./images/importance.png", bbox_inches="tight")
plt.show()
