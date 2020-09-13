import pickle
from scipy.stats import pearsonr
import pandas as pd

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + "aggregate_X.p", "rb") as f:
    X = pickle.load(f)

with open(data_loc + "_y_small.p", "rb") as f:
    y = pickle.load(f)

y = pd.Series(y, name="y")
print(len(y))
print(len(X))
X = X[0:len(y)]
df = pd.concat([X, y], axis=1)
print(df.head())

df_corr = df.corr()
#df_corr["abs_corr"] = abs(df_corr["y"])

df_corr.to_csv("./images/large_corr.csv")
