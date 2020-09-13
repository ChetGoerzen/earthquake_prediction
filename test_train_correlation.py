import pickle
from scipy.stats import pearsonr
import pandas as pd

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + "aggregate_X.p", "rb") as f:
    X = pickle.load(f)

with open(data_loc + "aggregate_y.p", "rb") as f:
    y = pickle.load(f)

y = pd.Series(y, name="y")

df = pd.concat([X, y], axis=1)
print(df.head())

df_corr = df.corr()

df_corr.to_csv("./images/corr.csv")
