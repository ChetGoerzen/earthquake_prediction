import pandas as pd
import numpy as np
from obspy import read, UTCDateTime
from scipy.stats import kurtosis, skew
from multiprocessing import pool
import pickle

net_info = pd.read_csv("./net_info.csv")
starttime = UTCDateTime(2016, 11, 5)

def make_features(X):
    
    strain = []
    strain.append(np.mean(X))
    strain.append(np.std(X))
    strain.append(kurtosis(X))
    strain.append(skew(X))
    
    return pd.Series(strain)

# from https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
def parallelize(df, func, n_cores):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    X = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return X

data_loc = "/mnt/readynas5/cgoerzen/tc2me/wfs/"

#for sta in net_info.sta:
# st = read(data_loc + "2016/11/05/20161105.5B." + str(sta) + "..DHZ.mseed")

# st = read(data_loc + "2016/11/05/20161105.5B.1107..DHZ.mseed")

sr = 500

catalog = pd.read_csv("../ToC2ME/Igonin/Catalog_Igonin2020_Beamforming.csv")
test_catalog = catalog[(catalog.Month==starttime.month) & (catalog.Day==starttime.day)]

event_times = []
for index, row in test_catalog.iterrows():
    tmp_time = 3600 * row.Hour + 60 * row.Minute + row.Second
    event_times.append(tmp_time)

marker_time = 0
event_idx = 0
time_to_failure = []
for i in range(int(max(event_times))):
    event = event_times[event_idx]
    ttf = event - marker_time
    time_to_failure.append(ttf)
    marker_time += 1
    if marker_time > event:
        event_idx +=1

pickle.dump(time_to_failure, open("./data/y.p", "wb"))

for sta in net_info.sta:
    print(sta)
    st = read(data_loc + "2016/11/05/20161105.5B." + str(sta) + "..DHZ.mseed")
    split_data = np.array_split(st[0].data, len(st[0].data) / int(sr))

    X = pd.DataFrame()
    #y = pd.Series()

    print(len(split_data))
    print(len(time_to_failure))

    ct = 0

    for data in split_data:

        if ct < len(time_to_failure):
        
            X_tmp = make_features(data)
            X = X.append(X_tmp, ignore_index=True)
            #y = y.append(pd.Series(time_to_failure[ct]))
        
        else:
            break

        ct += 1

    pickle.dump(X, open("./data/" + str(sta) + "_X.p", "wb"))
#pickle.dump(y, open("./data/y.p", "wb"))
