import pandas as pd
import numpy as np
from obspy import read, UTCDateTime
from scipy.stats import kurtosis, skew
from multiprocessing import pool
import pickle
from mpi4py import MPI
from multiprocessing import Pool
import time

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.Get_rank()
nprocs = comm.Get_size()

net_info = pd.read_csv("./net_info.csv")
output_dir = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"
data_loc = "/mnt/readynas5/cgoerzen/tc2me/wfs/"
starttime = UTCDateTime(2016, 11, 1)
sr = 500
catalog = pd.read_csv("../ToC2ME/Igonin/Catalog_Igonin2020_Beamforming.csv")

test_catalog = catalog[(catalog["Magnitude Mw"]  > 0.8) & (catalog["Month"] == 11) & (catalog.Day <= 20)]

def make_features(X):

    strain = []
    strain.append(np.mean(X))
    strain.append(np.std(X))
    strain.append(kurtosis(X))
    strain.append(skew(X))

    return pd.Series(strain)

# from https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
#def parallelize(df, func, n_cores):
#    df_split = np.array_split(df, n_cores)
#    pool = Pool(n_cores)
#    X = pd.concat(pool.map(func, df_split))
#    pool.close()
#    pool.join()
    
#    return X

if rank == 0:

    sendbuf = np.arange(0., 20.)

    # count: the size of each sub-task
    ave, res = divmod(sendbuf.size, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)

    # displacement: the starting index of each sub-task
    displ = [sum(count[:p]) for p in range(nprocs)]
    displ = np.array(displ)
else:
    sendbuf = None
    # initialize count on worker processes
    count = np.zeros(nprocs, dtype=np.int)
    displ = None

# broadcast count
comm.Bcast(count, root=0)

# initialize recvbuf on all processes
recvbuf = np.zeros(count[rank])

comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)


#for sta in net_info.sta:
# st = read(data_loc + "2016/11/05/20161105.5B." + str(sta) + "..DHZ.mseed")

# st = read(data_loc + "2016/11/05/20161105.5B.1107..DHZ.mseed")



def run(starttime, catalog, pool):
    print(starttime)
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

    pickle.dump(time_to_failure, open(output_dir + starttime.strftime("%Y%m%d") + "_y_small.p", "wb"))

#    for sta in net_info.sta:
#        print(sta)
#        st = read(data_loc + starttime.strftime("%Y/%m/%d/%Y%m%d")+ ".5B." + str(sta) + "..DHZ.mseed")
#        split_data = np.array_split(st[0].data, len(st[0].data) / int(sr))
#        
#        X = pd.DataFrame()
#
#        result = pool.map(make_features, split_data)
#        ct = 0
#
#
#        X = pd.concat(result, axis=1).transpose()[0:len(time_to_failure)]
#
#        pickle.dump(X, open(output_dir + starttime.strftime("%Y%m%d_") + str(sta) + "_X.p", "wb"))

dt = 3600 * 24 # Number of seconds in a day

#pool = Pool(8)

#for i in recvbuf:
#    
#    #st = time.time()
#    runner = run(starttime=starttime + i * dt, catalog=catalog, pool=pool)
#    #et = time.time()
#    #dt = et - st
#    #print(f"runtime: {dt:.3f}")
#    del runner

event_times = []
for index, row in test_catalog.iterrows():
    tmp_time = 3600 * 24 * row.Day + 3600 * row.Hour + 60 * row.Minute + row.Second
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

#pickle.dump(time_to_failure, open(output_dir + starttime.strftime("%Y%m%d") + "_y_small.p", "wb"))

pickle.dump(time_to_failure, open(output_dir + "_y_small.p", "wb"))
