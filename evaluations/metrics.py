import torch as tc 
import torch.nn
import torch.utils.data as Data 


def MAE(val,model,A,n_obs,n_pred,batch_size):
	#device = get_default_device()
	val_b, val_t, val_n, val_c = val.shape
	
	val_X = val[:,0:n_obs,:,:]
	val_Y = val[:,n_obs:,:,:]

	torch_val_data    = Data.TensorDataset(val_X, val_Y)
	val_loader        = Data.DataLoader(dataset= torch_val_data, batch_size=batch_size, shuffle=True)
	val_loader        = DeviceDataloader(val_loader, device)
	loss = tc.zeros(1).cuda()
	
	for step,(x,y) in enumerate(val_loader):
		
		y_b,y_t,y_n,y_c = y.shape
		model_input = x
		
		for i in range(n_pred):
			with tc.no_grad():
				model_output = model(model_input,A)
			
			if step == 0 and i == 0:
				loss = tc.abs(model_output - y[:,i,:,:]) #loss_dir[i]
			else:
				if loss.shape[0] == model_output.shape[0]:
					loss = loss + tc.abs(model_output - y[:,i,:,:]) #loss_dir[i]
				else:
					loss = tc.cat( [loss, tc.abs(model_output - y[:,i,:,:]) ], dim = 0)
			
			model_input = tc.cat( [ model_input[:,1:,:,:], model_output.unsqueeze(1)], dim=1 )
		
	loss = tc.sum(loss)/(val_b*n_pred*val_n*val_c)
	
	return loss
