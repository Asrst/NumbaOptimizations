
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse
import scipy as sp
import time
import NumbaOptimizations import NumbaGroupBy
import random
import os

# store data paths
data_paths = {}
for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        data_paths[filename] = os.path.join(dirname, filename)
        print(os.path.join(dirname, filename))


if __name__ == "__main__":

    train_df = pd.read_csv(data_paths['train_data.csv'])
    print(train_df.shape, train_df.columns)

    groupby_min = NumbaGroupBy(ops = 'min')
    groupby_max = NumbaGroupBy(ops = 'max')
    groupby_count = NumbaGroupBy(ops = 'count')
    groupby_size = NumbaGroupBy(ops = 'size')



    start = time.time()
    groupby_min = groupby_min.fit(train_df, index_col = 'patient_id')
    groupby_max = groupby_max.fit(train_df, index_col = 'patient_id')
    groupby_count = groupby_count.fit(train_df, index_col = 'patient_id')
    groupby_size = groupby_size.fit(train_df, index_col = 'patient_id')


    cdfs = []
    for col in ['event_name', 'specialty', 'plan_type']:
        cout = groupby_min.transform(train_df, value_col = col, fill_na = np.inf)
        cout = groupby_max.transform(train_df, value_col = col, fill_na = np.inf)
        cout = groupby_count.transform(train_df, value_col = col, fill_na = np.inf)
        cout = groupby_size.transform(train_df, value_col = col, fill_na = np.inf)
        cdfs.append(cout)

    res_arr = np.hstack(cdfs)
    end = time.time()
    print('time taken (in secs):', end-start)
    print(res_arr.shape)
