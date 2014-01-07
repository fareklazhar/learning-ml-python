import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
	""" performs the following: 
		lower casing the raw post in the preprocessing step
		extracting all individual words in the tokenization step
		converting all the words into the stemmed version

	"""
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()

		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
