import pandas as pd,numpy as np

def split_data(data,days,n_obs,n_pred,day_slot = 288,channel_in = 1):
    n = data.shape[1]
    day_batch_size = day_slot - (n_obs+n_pred) + 1
    temp_data = []
    for  i in range(days):
        day_batch_data = []
        for j in range(day_batch_size):
            day_batch_data.append( data[j:j+(n_obs+n_pred)] )
        temp_data.append(day_batch_data)
    return np.array(temp_data).reshape((days*day_batch_size,n_obs+n_pred,n,channel_in))


def data_gen(path, n_obs=12,n_pred=6):
    n_train, n_val, n_test = 34,5,5
    #data = pd.read_csv("../data/PeMS-M/V_228.csv",header=None).values
    data = pd.read_csv(path, header=None).values
    #print(data.shape,data.mean(axis=0),data.std(axis=0))
    #data = preprocessing.scale(data)
    #print(data.shape,data.mean(axis=0),data.std(axis=0))
    train = split_data(data[0:n_train*24*12,:],n_train,n_obs,n_pred)
    val = split_data(data[n_train*24*12:(n_train+n_val)*24*12 ,:],n_val,n_obs,n_pred)
    test = split_data(data[(n_train+n_val)*24*12:,:],n_test,n_obs,n_pred)

    return train,val,test

def scaled_adjacency(A,sigma2=0.1,e = 0.5):  
    n = A.shape[0]
    A = A/10000
    A2, A_mask = A*A , np.ones([n,n]) - np.identity(n)
    return np.exp(-A2/sigma2) * (np.exp(-A2/sigma2)>=e) * A_mask

