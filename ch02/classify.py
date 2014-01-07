from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np


# load the data
data = load_iris()
features = data['data']
features_names = data['feature_names']
target = data['target']
labels = data['target_names'][target]

#print features # just a list of features of each flower

for t, marker, c in zip(xrange(3), ">ox", "rgb"):
	# plot each class on its own to get different colours markers
	plt.scatter(features[target== t,0],
				features[target== t,1],
				marker=marker,
				c=c)
plt.grid()
plt.show()


# separating the flowers based on their petal length
plength = features[:,2]
is_setosa = (labels == 'setosa')
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print 'Max of Setosa: {0}'.format(max_setosa)
print 'Min of others: {0}'.format(min_non_setosa)
# this prints 1.9 and 3.0

# print features[:,2]
# if features[:,2] < 2: 
# 	print "Iris Setosa"
# else:
# 	"Iris Virginica or Iris Versicolour"

# retrieving all features that are non-setosa
features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica') # create array of labels that are only virginica

best_acc = -0.1
for fi in xrange(features.shape[1]):
	# all possible thresholds for this feature
	thresh = features[:, fi].copy()
	thresh.sort()
	# test threshold
	for t in thresh:
		print fi
		print features[:, fi]
		pred = (features[:, fi] > t)
		# the lines below select the best model for the data
		acc = (pred == virginica).mean()
		if acc > best_acc:
			best_acc = acc
			best_fit = fi
			best_t = t
print "best accuracy", best_acc
print "best fit", best_fit
print "best threshold", best_t