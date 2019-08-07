import torch as tc 
import torch.nn as nn
import torch.nn.functional as F

class GNL(nn.Module):
	"""docstring for GNL"""
	def __init__(self, channel_in, hidden_size,neighbor_agg_size):
		super(GNL, self).__init__()
		
		self.channel_in               = channel_in
		self.hidden_size              = hidden_size
		self.neighbor_agg_size        = neighbor_agg_size
		
		self.W_h = nn.Parameter(tc.zeros(size=(self.channel_in, self.hidden_size)))
		nn.init.xavier_uniform_(self.W_h.data, gain=1.414)

		self.W_alpha = nn.Parameter(tc.zeros(size=(self.hidden_size, self.neighbor_agg_size)))
		nn.init.xavier_uniform_(self.W_alpha.data, gain=1.414)

		self.att_input_weight_vetor  =  nn.Parameter(tc.zeros(size=(self.neighbor_agg_size*2,1)))
		nn.init.xavier_uniform_(self.att_input_weight_vetor.data, gain=1.414)

		self.leakyrelu  = nn.LeakyReLU(0.2)

		self.W_f = nn.Parameter(tc.zeros(size=(self.channel_in+self.hidden_size+self.neighbor_agg_size, self.neighbor_agg_size)))
		nn.init.xavier_uniform_(self.W_f.data, gain=1.414)

		self.W_e = nn.Parameter(tc.zeros(size=(self.channel_in+self.hidden_size+self.neighbor_agg_size, self.hidden_size)))
		nn.init.xavier_uniform_(self.W_f.data, gain=1.414)

		self.W_g = nn.Parameter(tc.zeros(size=(self.channel_in+self.hidden_size+self.neighbor_agg_size, self.hidden_size)))
		nn.init.xavier_uniform_(self.W_f.data, gain=1.414)

		self.W_r = nn.Parameter(tc.zeros(size=(self.channel_in+self.hidden_size+self.neighbor_agg_size, self.hidden_size)))
		nn.init.xavier_uniform_(self.W_f.data, gain=1.414)

		self.W_u = nn.Parameter(tc.zeros(size=(self.channel_in+self.hidden_size+self.neighbor_agg_size, self.hidden_size)))
		nn.init.xavier_uniform_(self.W_f.data, gain=1.414)

		self.predict_fc = nn.Linear(self.hidden_size,self.channel_in)
	def forward(self,X,A):
		b, t, n, c_in = X.shape

		A = A.squeeze(0).repeat(b,1,1)

		#init_Hidden = tc.zeros(n,self.hidden_size)
		Hidden  = tc.matmul(X[:,0,:,:],self.W_h)
		#print(Hidden.shape)
		for time_step in range(t):
			#########################################################################################
			Hidden_z  = tc.matmul(Hidden,self.W_alpha)  
			
			att_input_cat = tc.cat( [ Hidden_z.repeat(1,1,n).view(b,n*n,-1), Hidden_z.repeat(1,n,1) ], dim=2)#.view(n,-1,self.neighbor_agg_size*2)
			e             = self.leakyrelu(  tc.matmul(att_input_cat,self.att_input_weight_vetor ).squeeze(-1).view(-1,n,n)  )
			zero_vec      = -9e15*tc.ones_like(e)
			#print(e.shape)
			att_e         = tc.where(A>0,e,zero_vec)
			att_e         = F.softmax(att_e, dim =2) 
			att_e         = F.dropout(att_e, 0.2, training=self.training)
			Z             = tc.bmm(att_e, Hidden_z)
			#########################################################################################
			X_Z_H_cat     = tc.cat([X[:,time_step,:,:],Z,Hidden],dim=2) 
			
			Z_hat         = tc.sigmoid(tc.matmul(X_Z_H_cat ,self.W_f)) * Z
			H_hat         = tc.sigmoid(tc.matmul(X_Z_H_cat ,self.W_e)) * Hidden

			X_Zh_Hh_cat   = tc.cat([X[:,time_step,:,:],Z_hat,H_hat],dim=2) 
			X_Z_Hh_cat    = tc.cat([X[:,time_step,:,:],Z,H_hat],dim=2)
			X_Zh_H_cat    = tc.cat([X[:,time_step,:,:],Z_hat,Hidden],dim=2) 

			g             = tc.sigmoid(tc.matmul(X_Z_H_cat ,self.W_g))
			r             = tc.sigmoid(tc.matmul(X_Z_H_cat ,self.W_r))
			
			Hidden        = g * r * tc.tanh(tc.matmul(X_Zh_Hh_cat,self.W_u))    + \
						    (1-g) * r * tc.tanh(tc.matmul(X_Z_Hh_cat,self.W_u)) + \
						    g * (1-r) * tc.tanh(tc.matmul(X_Zh_H_cat,self.W_u)) + \
						    (1-g) * (1-r) * tc.tanh(tc.matmul(X_Z_H_cat,self.W_u)) 

		return  self.predict_fc(Hidden.view(-1,self.hidden_size)).view(b,n,self.channel_in)
