import torch as tc 
import torch.nn
import torch.utils.data as Data 
from torch.autograd import Variable

def bulid_model(channel_in,hidden_size,neighbor_agg_size):
  return GNL(channel_in,hidden_size,neighbor_agg_size)
  
def train_model(train,val,model,A,hyper_para,time_slag):
	epoches, batch_size, opti, early_stop_times, device = hyper_para
	n_obs, n_pred       = time_slag
	min_mae = float("inf")
	model = to_device(model, device)

	train_data     = Data.TensorDataset(train[:,0:n_obs,:,:], train[:,n_obs:,:,:])
	train_loader   = Data.DataLoader(dataset= train_data, batch_size=batch_size, shuffle=True)
	train_loader   = DeviceDataloader(train_loader, device)

	if opti == "RMSprop":
		opt_model = tc.optim.RMSprop( model.parameters() )
	if opti == "Adam":
		opt_model = tc.optim.Adam( model.parameters() )
		
	for epoch in range(epoches):
		for step,(x,y) in enumerate(train_loader):
			y_b,y_t,y_n,y_c = y.shape
			model_input = x
			for i in range(n_pred):
				model_output = model(model_input,A)
				if i == 0:
					loss = (model_output - y[:,i,:,:])**2 #loss_dir[i]
				else:
					loss = loss + (model_output - y[:,i,:,:])**2 #loss_dir[i]

				model_input = tc.cat( [ model_input[:,1:,:,:], model_output.unsqueeze(1)], dim=1 )
			loss = tc.sum(loss)/(y_b*y_t*y_n*y_c)

			opt_model.zero_grad()
			loss.backward()
			opt_model.step()
			print(f"epoch: {epoch}, setp: {step}, loss: {loss.data}")
		
		mae = MAE(val,gnl,A,n_obs,n_pred,batch_size)
		###################################################################
		cur_mae = mae.item()              #early_stop
		if cur_mae < min_mae:
			min_mae = cur_mae
			count = 0
		else:
			count += 1
			if count > early_stop_times:
				break
		#####################################################################
		print(f"epoch: {epoch}, MAE: {mae.data}")
	return model
