import string
import unicodedata
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
import h5py
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import numpy as np
from sys import argv


# ---------- Data Loader
all_letters = string.ascii_letters + '.,; '
def unicode_to_ascii(txt):
	return ''.join([c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn' and c in all_letters])


def read_lines(file_name):
	lines = open(file_name, encoding='utf-8').read().strip().split('\n')
	lines = ''.join([unicode_to_ascii(line) for line in lines])
	return lines


def load_txt(dataset, all_lines, all_categories	):
	for cat in listdir("data/wiki-"+dataset):
		lines = read_lines("data/wiki-"+dataset + "/" + cat)
		all_lines.append(lines)
		all_categories[cat] = lines


# ---------- TF-IDF
def compute_tfidf(dataset):
	all_categories = {}
	all_lines = []
	all_tfidfs = {}
	load_txt(dataset, all_lines, all_categories)
	
	# ----- TF-IDF
	vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
	vectorizer.fit(all_lines)
	for category in all_categories:
		vector = vectorizer.transform([all_categories[category]])
		vector = vector.toarray()[0]	
		all_tfidfs[category] = vector

	all_cls = listdir("data/wiki-"+dataset)
	vectors = []
	for cat in all_cls:
		vectors.append(all_tfidfs[cat])
	print(len(vectors[1]))
	# ----- HDF5
	with h5py.File('outputs/wiki_tfidf_' + dataset + '.hdf5', 'w') as feat_file:
		feat_file.create_dataset('cls_embedding', data=vectors)
	
	# ----- t-SNE visualization of the extracted tf-idf vectors
	tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=5000)
	results = tsne.fit_transform(vectors)
	plots = []
	for i in range(len(results)):
	    res = results[i]
	    res = np.array(res)
	    pl = go.Scatter(x=[res[0]], y=[res[1]], mode='markers',
	                    marker=dict(size=10, color=i, colorscale='Jet', opacity=0.8), name=all_cls[i])
	    plots.append(pl)

	py.plot(plots, filename='outputs/tsne_tfidf_' + dataset + '.html', auto_open=False)	
	


def main():
	dataset = argv[1]
	print('Dataset: ' +  dataset)
	compute_tfidf(dataset)

if __name__  == "__main__":
	main()