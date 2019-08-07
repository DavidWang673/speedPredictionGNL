import pandas as pd, numpy as np, torch as tc

n_obs, n_pred =6, 3

data_path = "drive/My Drive/Projects/data/V_228.csv"
train,val,test = data_gen(data_path)

train  = tc.from_numpy(train).float()
val    = tc.from_numpy(val).float()
test   = tc.from_numpy(test).float()

node_path = "drive/My Drive/Projects/data/W_228.csv" 
A = pd.read_csv(node_path, header=None).values
A = scaled_adjacency(A)
A = tc.from_numpy(A).float().cuda()


device = get_default_device()

gnl = bulid_model( channel_in=1, hidden_size=8,neighbor_agg_size=8)
#gnl = to_device(gnl,device)

epoches, batch_size, opti, early_stop_times = 100, 80, "RMSprop", 20
hyper_para = [epoches, batch_size, opti, early_stop_times,device]

model = train_model(train,val,gnl,A,hyper_para,[n_obs, n_pred])
tc.save(model,"drive/My Drive/Projects/model/GNL.pkl")
