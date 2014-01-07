import os
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1) # minimum document frequency

print vectorizer

content = ["how to format my hard disk", "Hard disk format problems"]
X = vectorizer.fit_transform(content)
print vectorizer.get_feature_names()
print X.toarray().transpose()

DIR = "/learning-ml-python/ch03/data/toy"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape

print("#samples: %d, #features: %d" % (num_samples, num_features)) #samples: 5, #features: 25