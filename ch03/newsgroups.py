import sklearn.datasets
from stemming import StemmedTfidfVectorizer
import scipy as sp
from sklearn.cluster import KMeans

MLCOMP_DIR = 'data/'

dataset = sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root=MLCOMP_DIR)

print dataset.filenames
print len(dataset.filenames)
print dataset.target_names

groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR, categories=groups)
test_data = sklearn.datasets.load_mlcomp("20news-18828", "test", mlcomp_root=MLCOMP_DIR, categories=groups)

print train_data
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
# using all data
# vectorized = vectorizer.fit_transform(dataset.data)

# using train_data
vectorized = vectorizer.fit_transform(train_data.data)

# using test_data
vectorized = vectorizer.fit_transform(test_data.data)

num_samples, num_features = vectorized.shape

print "#Samples: {0}, #Features: {1}".format(num_samples, num_features)


num_clusters = 50
km = KMeans(n_clusters=num_clusters, init="random", n_init=1, verbose=1)
km.fit(vectorized)

new_post = "Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks."
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

# At this point, now that we have all the clustering, we do not need to compare
# new_post_vec to ALL the post vectors, just posts in the same cluster

similar_indices = (km.labels_==new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
	dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
	similar.append((dist, train_data.data[i]))

similar = sorted(similar)

print len(similar) # this is the number of posts in the cluster of our post
print "Most similar: ", similar[0][1]
