import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_error as mae

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + "model.p", "rb") as f:
    model = pickle.load(f)

with open(data_loc + "aggregate_X.p", "rb") as f:
    X = pd.DataFrame(pickle.load(f))

with open(data_loc + "y_large.p", "rb") as f:
    y = pd.Series(pickle.load(f))

y = y[0:len(X)]

stop = int(len(X) * 0.85)

X_train = X[0:stop]
y_train = y[0:stop]
X_test = X[stop:-1]
y_test = y[stop:-1]
#D_train = xgb.DMatrix(X_train, label=Y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

fig, ax = plt.subplots(dpi=500)
ax.plot(train_preds, "r")
ax.plot(y_train, "b")
plt.xlim(0, 1000)
plt.savefig("./images/fit_to_train.png", bbox_inches='tight', dpi=500)
plt.show()

fig, ax = plt.subplots(dpi=500)
ax.plot(test_preds, "r")
ax.plot(y_test, "b")
plt.xlim(0, 1000)
plt.savefig("./images/fit_to_test.png", bbox_inches='tight', dpi=500)
plt.show()

#print(train_preds[0:10])
#print(Y_train[0:10])
print(mae(test_preds, y_test))
fig, ax = plt.subplots()
xgb.plot_importance(model, ax=ax, max_num_features=25)
plt.savefig("./images/importance.png", bbox_inches="tight")
plt.show()

pickle.dump(test_preds, open("./images/preds.p", "wb"))
pickle.dump(y_test, open("./images/y_test.p", "wb"))
