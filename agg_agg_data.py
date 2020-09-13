import numpy as np
import pickle
import os
import pandas as pd

data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

test_data = []
y_data = []

for file in os.listdir(data_loc):
    if file.split("_")[-1] == "df.p":
        with open(data_loc + file, "rb") as f:
            df = pickle.load(f)
        test_data.append(df)

    if file.split("_")[-1] == "y.p":
        with open(data_loc + file, "rb") as f:
            y = pickle.load(f)
        y_data.append(pd.Series(y))

X = pd.concat(test_data, ignore_index=True)
y = pd.concat(y_data, ignore_index=True)

pickle.dump(X, open(data_loc + "aggregate_X.p", "wb"))
pickle.dump(y, open(data_loc + "aggregate_y.p", "wb"))
