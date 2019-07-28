from __future__ import division
from AdaptiveBinning import AdaptiveBinning
import numpy as np

def softmax(x):
    """Compute softmax for x on axis 1."""
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

# raw_output.npy simply save the raw ouput before softmax of a network. Its shape is (m,n) where m is the numner of samples and n is the number of classes.
file_name = 'raw_output.npy'

data = np.load(file_name)

probability = softmax(data[:, 1:])
prediction = np.argmax(probability, axis=1)
label = data[:, 0]

infer_results = []

for i in range(len(data)):
	correctness = (label[i] == prediction[i])
	infer_results.append([probability[i][prediction[i]], correctness])

# Call AdaptiveBinning.
AECE, AMCE, confidence, accuracy, cof_min, cof_max = AdaptiveBinning(infer_results, True)

print('ECE based on adaptive binning: {}'.format(AECE))
print('MCE based on adaptive binning: {}'.format(AMCE))


