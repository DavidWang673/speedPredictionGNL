import torch as tc
class DeviceDataloader():
	"""docstring for DeviceDataloader"""
	def __init__(self, dl, device):
		self.dl = dl
		self.device = device

	def __iter__(self):
		for b in self.dl:
			yield to_device(b, self.device)

	def __len__(self):
		return len(self.dl)


def get_default_device():
	if tc.cuda.is_available():
		return tc.device("cuda")
	else:
		return tc.device("cpu")

def to_device(data, device):
	if  isinstance(data, (list,tuple)):
		return [to_device(x, device) for x in data]

	return data.to(device,non_blocking=True)
