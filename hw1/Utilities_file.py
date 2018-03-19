import numpy as np

def shuffle(x, y):
	N = x.shape[0]
	shuffled_indexes = np.arange(N)
	np.random.shuffle(shuffled_indexes)
	shuffled_x = np.zeros(x.shape)
	shuffled_y = np.zeros(y.shape)
	shuffled_x[np.arange(N), :] = x[shuffled_indexes, :]
	shuffled_y[np.arange(N), :] = y[shuffled_indexes, :]
	return shuffled_x, shuffled_y

def getBatches(x, y, batch_size):
	batches = []
	s_x, s_y = shuffle(x, y)
	splits = int(s_x.shape[0]/batch_size)
	for i in range(splits):
		batches.append((s_x[i*batch_size:(i+1)*batch_size, :], s_y[i*batch_size:(i+1)*batch_size, :]))
	return batches