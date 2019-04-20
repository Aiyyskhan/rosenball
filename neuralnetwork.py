import numpy as np

'''

	*** example ***

	INIT_WEIGHT_MAX = 0.5
	NEURON_LAYERS = [2, 3, 2]
	
	nn = NeuralNetwork(NEURON_LAYERS, INIT_WEIGHT_MAX)
	data = np.array([1,2])
	res = nn(data)
	print(res)

'''

def nonlin(s, deriv = False):
	if deriv == True:
		return s * (1 - s)
	return 1 / (1 + np.exp(-s))

class NeuralNetwork:
	def __init__(self, layers_list, weight_max):
		self.syn_list = []
		for i in range(len(layers_list)-1):
			syn_arr = self._weight_initializer(layers_list[i], layers_list[i+1], weight_max)
			self.syn_list.append(syn_arr)

	def __call__(self, data):
		data_array = data
		for syn_arr in self.syn_list:
			data_array = self._calculate(data_array, syn_arr)
		return data_array

	def _weight_initializer(self, pre, post, w_max):
		return np.random.normal(0.0, pow(post, -w_max), (pre + 1, post))

	def _calculate(self, data_arr, syn_arr):
		return nonlin(np.dot(data_arr, syn_arr[:-1]) + syn_arr[-1])

