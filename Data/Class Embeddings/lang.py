

class Lang:
	"""Class Language
	"""
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {} #{0:"SOS", 1:"EOS"}
		self.n_words = 0 # 2

	def add_sentence(self, sentence):
		for word in sentence.split(' '):
			self.add_word(word)

	def add_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1