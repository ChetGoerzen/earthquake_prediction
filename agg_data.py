import pandas as pd
import pickle
import os
from obspy import UTCDateTime

starttime = UTCDateTime(2016, 11, 6)
date_str = starttime.strftime("%Y%m%d")
data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"
#print(date_str)
to_concat = []
for file in os.listdir(data_loc):
    if file.split("_")[0] == date_str:
        if file.split("_")[-1] == "X.p":
            sta = file.split("_")[1]
            columns = ["mean_" + str(sta), "std_" + str(sta), "kur_" + str(sta), "skew_" + str(sta)]
            with open(data_loc + file, "rb") as f:
                X = pickle.load(f)
            X.columns = columns
            to_concat.append(X)

##
#    if file.split("_")[-1] == "X.p":
#        if len(file.split("_")) == 2:
#            sta = file.split("_")[0]
#            columns = ["mean_" + str(sta), "std_" + str(sta), "kur_" + str(sta), "skew_" + str(sta)]
#            with open(data_loc + file, "rb") as f:
#                X = pickle.load(f)
#            X.columns = columns
#            to_concat.append(X)

df = pd.concat(to_concat, axis=1)
df.columns = sorted(df.columns)
print(df)
pickle.dump(df, open(data_loc + date_str + "_df.p", "wb"))
