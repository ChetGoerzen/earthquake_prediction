import pickle
from obspy import UTCDateTime
import xgboost as xgb
from sklearn.metrics import mean_absolute_error as mae

starttime = UTCDateTime(2016, 11, 6)
date_str = starttime.strftime("%Y%m%d")
data_loc = "/mnt/readynas5/cgoerzen/earthquake_prediction/data/"

with open(data_loc + date_str + "_df.p", "rb") as f:
    X = pickle.load(f)

with open(data_loc + date_str + "_y.p", "rb") as f:
    y = pickle.load(f)

with open(data_loc + "model.p", "rb") as f:
    model = pickle.load(f)

D_test = xgb.DMatrix(X, label=y)

preds = model.predict(D_test)

print(mae(preds, y))

pickle.dump(preds, open("./images/test_preds.p", "wb"))
pickle.dump(y, open("./images/test_y.p", "wb"))
