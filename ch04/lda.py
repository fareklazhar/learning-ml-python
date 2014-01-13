import numpy as np
from scipy.spatial import distance
from gensim import corpora, models, similarities

corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1)
# up to this point, we have built a topic model


topics = [model[c] for c in corpus]
dense = np.zeros( (len(topics), 100), float )

for ti,t in enumerate(topics):
	for tj, v in t:
			dense[ti, tj] = v

print dense

pairwise = distance.squareform(distance.pdist(dense))
largest = pairwise.max()

for ti in range(len(topics)):
	pairwise[ti,ti] = largest + 1

def closest_to(doc_id):
	return pairwise[doc_id].argmin()
