import os
import sys
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1, stop_words="english") # minimum document frequency

print vectorizer

content = ["how to format my hard disk", "Hard disk format problems"]
X = vectorizer.fit_transform(content)
print vectorizer.get_feature_names()
print X.toarray().transpose()


DIR = "/home/tom/learning-ml-python/ch03/data/toy"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape

print("#samples: %d, #features: %d" % (num_samples, num_features)) #samples: 5, #features: 25
print vectorizer.get_feature_names()
print "With {0} different words".format(len(vectorizer.get_feature_names()))

# goal here is to find the most similar post for the post new_post defined below
new_post = "imaging database"
new_post_vec = vectorizer.transform([new_post])
new_post_array = new_post_vec.toarray() # need it to be the full array if we want to do similarity calculations

# calculate Euclidean distance, of new post with the old posts
def dist_raw(v1, v2):
	delta = v1 - v2
	return sp.linalg.norm(delta.toarray())

# calculate the vector distance
def dist_norm(v1, v2):
	v1_normalized = v1 / sp.linalg.norm(v1.toarray())
	v2_normalized = v2 / sp.linalg.norm(v2.toarray())
	delta = v1_normalized - v2_normalized
	return sp.linalg.norm(delta.toarray())


best_doc = None
best_dist = sys.maxint
best_i = None

print

for i in range(0, num_samples):
	post = posts[i] # iterate through the posts
	if post == new_post:
		continue
	post_vec = X_train.getrow(i)
	# d = dist_raw(post_vec, new_post_vec)
	d = dist_norm(post_vec, new_post_vec)
	print "=== Post {0} with dist={1}: {2}".format(i, d, post)
	if d < best_dist:
		best_doc = post
		best_dist = d
		best_i = i
# this is before normalizing.
print "The best post is {0}: '{1}' with dist={2}".format(best_i, best_doc, best_dist)


