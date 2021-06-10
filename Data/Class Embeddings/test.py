import random
import torch
import torch.nn as nn
from torch import optim
from os import listdir
from os.path import join
import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import operator
import h5py
from argparse import ArgumentParser
from models import LSTM, GRU
from data_handler import DataHandler
import fastText


class TextEmbedder():
	def __init__(self, opt):
		self.data_handler = DataHandler(opt)	
		self._load_fastText_model(opt.fastText_model_path)
		self.dataset = opt.dataset
		self.method = opt.method
		self.model = opt.model
		if self.method == "rnn" or "rnn-mean":	
			self.hidden_size = opt.hidden_size
			self._load_rnn(self.model, opt.checkpoint_path)

	def _load_rnn(self, model, checkpoint_path):
		"""Loads rnn model if method is rnn/rnn-mean
		"""
		fastText_emb_size = 300
		num_cls = self.data_handler.num_cls
		if model == 'gru':
			self.rnn_model = GRU(fastText_emb_size, self.hidden_size, num_cls)	
		elif model == 'lstm':
			self.rnn_model = LSTM(fastText_emb_size, self.hidden_size, num_cls)
		self.rnn_model.load_state_dict(torch.load(checkpoint_path))

	def _load_fastText_model(self, ft_model_path):
		"""Loads fastText model
		"""
		print("Loading fastText model ...")
		self.ft_model = fastText.load_model(ft_model_path)	
		print("Loaded fastText")

	def _get_word_tensors_from_sentence(self, sentence):
		"""Returns list of fastText vectors for words of a given sentence
		"""
		embds = [self.ft_model.get_word_vector(word) for word in sentence.split(' ')]
		return embds

	def _get_whole_sentence_ft_tensor(self, sentence):
		"""Returns fastText vector for a whole sentence (without RNN)
		"""
		return self.ft_model.get_sentence_vector(sentence)

	def _get_joined_sent_emb_rnn(self, cat):
		"""Returns ft-embedded vector for joined sentences of a category
		"""
		cat_sentences = ' '.join(self.data_handler.all_sentences[cat])
		return torch.tensor(self._get_word_tensors_from_sentence(cat_sentences))

	def _get_single_sent_emb_rnn(self, sent):
		"""Returns ft-embedded vector for a given sentence
		"""
		return torch.tensor(self._get_word_tensors_from_sentence(sent))

	def _get_direct_ft_cat(self, cat):
		"""Returns fastText vector for joined senteces of a category
		"""
		cat_sentences = ' '.join(self.data_handler.all_sentences[cat])
		return self._get_whole_sentence_ft_tensor(cat_sentences)

	def _get_direct_ft_name(self, cat):
		if self.dataset =="awa2":
			revised_name = cat.replace("+", " ")
			return self.ft_model.get_word_vector(revised_name)
		elif self.dataset == "cub":
			revised_name = cat.split(".")[1].lower().replace("_", " ")
			return self.ft_model.get_word_vector(revised_name)

	def get_emb_vec(self, cat):
		"""Returns embedding vector for sentences of a given category
		"""
		if self.method == "fastText-context":
			ft_emb = self._get_direct_ft_cat(cat)
			return ft_emb
		elif self.method == "fastText-names":
			ft_emb = self._get_direct_ft_name(cat)
			return ft_emb
		elif self.method == "rnn":	
			sent = self._get_joined_sent_emb_rnn(cat)
			output, hidden = self.rnn_model(sent.unsqueeze(1))
			return hidden.view(self.hidden_size).detach().numpy()
		elif self.method == "rnn-mean":
			vecs = []
			hidden0 = torch.zeros(1, 1, self.hidden_size)
			for s in self.data_handler.all_sentences[cat]:
				sent = self._get_single_sent_emb_rnn(s)
				_, hidden = self.rnn_model(sent.unsqueeze(1), hidden0)
				vecs.append(hidden.view(self.hidden_size).detach().numpy())
			return np.mean(vecs, axis=0)

	def generate_hdf(self):
		"""Creates hdf5 output for the claculated class embeddings
		"""
		print("generating hdf file ...")
		all_classes = []
		for l in open("data/allclasses-"+self.dataset+".txt","r").readlines():
			all_classes.append(l.strip())
		vecs = []
		for cat in all_classes:
			vecs.append(self.get_emb_vec(cat))
		file_name = "outputs/" + self.method + "_" + self.dataset + "_" + self.model + ".hdf5"
		with h5py.File(file_name, 'w') as f:
			f.create_dataset('cls_embedding', data=np.array(vecs))

	def tsne_vis(self, use_means=True):
		"""Creates t-SNE visualization of the generated embeddings
		"""
		vecs = []
		for cat in self.data_handler.all_categories:
			vecs.append(self.get_emb_vec(cat))

		tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=6000)
		results = tsne.fit_transform(vecs)

		plots = []
		for c in range(len(self.data_handler.all_categories)):
		    pl = go.Scatter(x=[results[c, 0]], y=[results[c, 1]], mode='markers+text',text=[self.data_handler.all_categories[c]],
		                    marker=dict(size=10, color=c, colorscale='Jet', opacity=0.8), name=self.data_handler.all_categories[c],
		                    textfont=dict(size=14,),textposition='bottom center')
		    plots.append(pl)  
		file_name = "outputs/tsne_" + self.method + "_" + self.dataset + "_" + self.model + ".html"
		py.plot(plots, filename=file_name, auto_open=True)
		

def main(opt):
	tester = TextEmbedder(opt)
	if opt.hdf == "yes":
		tester.generate_hdf()
	if opt.tsne == "yes":
		tester.tsne_vis()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fastText-model-path', type=str, default="")
    parser.add_argument('--dataset', type=str, default="awa2")
    parser.add_argument('--model', type=str, default="lstm")
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--method', type=str, default="rnn")
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--hdf', type=str, default="yes")
    parser.add_argument('--tsne', type=str, default="yes")
    opt = parser.parse_args()
    main(opt)
