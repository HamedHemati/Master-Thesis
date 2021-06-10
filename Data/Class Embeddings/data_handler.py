from os import listdir
from os.path import join
import unicodedata
import string
import re
import random
import operator
import torch
import fastText
from torch.utils.data import Dataset, DataLoader
from lang import Lang


class TextDS(Dataset):
	def __init__(self, all_sentences, all_labels, ft_model_path):
		super(TextDS, self).__init__()
		self.all_sentences = all_sentences
		self.all_labels = all_labels
		self._load_fastText_model(ft_model_path)

	def _load_fastText_model(self, ft_model_path):
		print("Loading fastText model ...")
		self.ft_model = fastText.load_model(ft_model_path)	
		print("Loaded fastText")

	def _get_word_tensors_from_sentence(self, sentence):
		embds = [self.ft_model.get_word_vector(word) for word in sentence.split(' ')]
		return embds

	def __getitem__(self, idx):
		sent = self.all_sentences[idx]
		sent = self._get_word_tensors_from_sentence(sent)
		return torch.tensor(sent), torch.tensor([self.all_labels[idx]])

	def __len__(self):
		return len(self.all_labels)		


class DataHandler():
	def __init__(self, opt):
		self.opt = opt
		self._prepare_data(opt.dataset)

	def _unicode_to_ascii(self, s):
	    return ''.join(
	        c for c in unicodedata.normalize('NFD', s)
	        if unicodedata.category(c) != 'Mn')

	def _normalize_string(self, s):
		# Lowercase, trim, and remove non-letter characters
	    s = self._unicode_to_ascii(s.lower().strip())
	    s = re.sub(r"([.!?])", r" \1", s)
	    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	    return s

	def _prepare_data(self, dataset):
		self.lang = Lang(dataset)
		print("Loading and preparing dataset ...")
		path = 'data/wiki-'+ dataset + '/'
		self.all_sentences_list = []
		self.all_labels_list = []
		self.all_sentences = {}
		self.all_categories = []
		all_cats = listdir(path)
		self.all_categories.extend(all_cats)
		self.num_cls = len(all_cats)
		cls_idx = 0
		for cat in all_cats:
			lines = open(join(path,cat), encoding='utf-8').read().strip().split('\n')
			lines = [self._normalize_string(l) for l in lines if len(l)>20]
			self.all_sentences_list.extend(lines)
			self.all_sentences[cat] = lines
			self.all_labels_list.extend([cls_idx]*len(lines))
			for l in lines:
				self.lang.add_sentence(l)
			cls_idx += 1

	def get_dataloader(self):
		ds = TextDS(self.all_sentences_list, self.all_labels_list, self.opt.fastText_model_path)
		return DataLoader(dataset=ds, batch_size=1, shuffle=True)