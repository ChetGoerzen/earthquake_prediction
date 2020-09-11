import pandas as pd
import pickle
import os

#for file in os.listdir("./data"):

# = pd.DataFrame()

#with open("./data/1107_X.p", "rb") as f:
#    X1 = pickle.load(f)

#with open("./data/1215_X.p", "rb") as f:
#    X2 = pickle.load(f)

#r1 = pd.merge(X1, X2)

#print(r1.head(10))

to_concat = []

for file in os.listdir("./data/"):
    if file.split("_")[-1] == "X.p":
        sta = file.split("_")[0]
        columns = ["mean_" + str(sta), "std_" + str(sta), "kur_" + str(sta), "skew_" + str(sta)]
        with open("./data/" + file, "rb") as f:
            X = pickle.load(f)
        X.columns = columns
        to_concat.append(X)

df = pd.concat(to_concat, axis=1)

pickle.dump(df, open("./data/df.p", "wb"))
