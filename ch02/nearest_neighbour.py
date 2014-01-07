import numpy as np

def distance(p0, p1):
	return np.sum((p0-p1)**2)

# when classifying, look at the dataset for the point closest 
# then look at the label

def nn_classify(training_set, training_label, new_example):
	distances = np.array([distance(t, new_example) for t in training_set])
	nearest = distances.argmin()
	return training_label[nearest]