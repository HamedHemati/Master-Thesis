import torch
import numpy as np
from os.path import join
from os import listdir
import random
import json
from .plotter import Plotter
from .data_handler import DataHandler
from .helper_functions import read_param_file, get_new_labels
from .zsl import ZSL
from .gan_functions import generate_synthetic_features, generate_synthetic_features_mean
from models.zswgan import NetG_ZSWGAN
from models.filmwgan import NetG_FiLMWGAN
from models.wgan import NetG_WGAN
from models.gan import NetG_GAN


class Tester():
	def __init__(self, opt, cuda=True):
		self.opt = opt
		self.cuda = cuda
		read_param_file(self.opt)
		self.data_handler = DataHandler(opt)
		self._init_netg()
		self.plotter = Plotter(opt.path)
		self.opt.images_path = self.opt.images_path + '/' + self.opt.dataset + '/images'

	def _init_netg(self):
		if self.opt.model == 'ZSWGAN':
			self.net_g = NetG_ZSWGAN(self.opt)
		elif self.opt.model == 'FiLMWGAN':
			self.net_g = NetG_FiLMWGAN(self.opt)
		elif self.opt.model == 'GAN':
			self.net_g = NetG_GAN(self.opt)
		elif self.opt.model == 'WGAN':
			self.net_g = NetG_WGAN(self.opt)		
		self.net_g.eval()	
		if self.cuda:
			self.net_g.cuda()

	def _load_checkpoint(self, use_gzsl_checkpoint):
		checkpoint_path = self.opt.path + "/net_g_checkpoints/"
		if use_gzsl_checkpoint:
			checkpoint_path += "gzsl_best.pth"
		else:
			checkpoint_path += "zsl_best.pth"
		self.net_g.load_state_dict(torch.load(checkpoint_path))
		print("Loaded the checkpoint")

	def run_zsl(self, compute_confusion=False):
		"""Standard Zero-Shot Learnig
		"""
		self._load_checkpoint(use_gzsl_checkpoint=False)
		n_samples = self.opt.n_synth_samples
		syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		n_cls = self.data_handler.n_test_classes
		Y = get_new_labels(syn_labels, self.data_handler.unseen_classes)
		zsl = ZSL(syn_features, Y, self.data_handler, n_samples, 25, n_cls, 0.001, self.opt.classifier_type)
		acc_seen_zsl = zsl.run_zsl(compute_confusion)
		print("-"*30)
		print('Acc. ZSL: %.4f%%\n' % (acc_seen_zsl))
		if compute_confusion:
			confusion_matrix = zsl.confusion_matrix
			test_cls = self.data_handler.unseen_classes
			confusion_list_x = [self.data_handler.class_names[c] for c in test_cls]
			confusion_list_y = [self.data_handler.class_names[c] for c in test_cls]
			self.plotter.plot_confusion_matrix(confusion_list_x, confusion_list_y, confusion_matrix, file_name='confusion-matrix-zsl.html')	
		
		output_text = "ZSC - " + self.opt.classifier_type + " - " + str(self.opt.n_synth_samples) + ":\n"
		output_text += str(acc_seen_zsl) + "\n\n"
		with open(join(self.opt.path ,"zsc_output.txt"), "a") as file:
			file.write(output_text)

	def run_gzsl(self, compute_confusion=False):
		"""Generalized Zero-Shot Learning
		"""
		self._load_checkpoint(use_gzsl_checkpoint=True)
		n_samples = self.opt.n_synth_samples
		syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		X = torch.cat([self.data_handler.train_features, syn_features], dim=0)
		Y = torch.cat([self.data_handler.train_labels, syn_labels], dim=0)
		n_cls = self.data_handler.n_train_classes + self.data_handler.n_test_classes
		gzsl = ZSL(X, Y, self.data_handler, n_samples, 25, n_cls, 0.001, self.opt.classifier_type)
		acc_seen, acc_unseen, acc_h = gzsl.run_gzsl(compute_confusion)
		print("-" * 30)
		print('Acc. Unseen: %.4f%%\nAcc. Seen: %.4f%%\nAcc. H: %.4f%%\n\n' % (acc_unseen, acc_seen, acc_h))
		if compute_confusion:
			confusion_matrix = gzsl.confusion_matrix 
			n_test_cls = self.data_handler.seen_classes.size(0) + self.data_handler.unseen_classes.size(0)
			confusion_list_x = [self.data_handler.class_names[c] for c in range(n_test_cls)]
			confusion_list_y = [self.data_handler.class_names[c] for c in range(n_test_cls)]
			self.plotter.plot_confusion_matrix(confusion_list_x, confusion_list_y, confusion_matrix, file_name='confusion-matrix-gzsl.html')

		output_text = "GZSC - " + self.opt.classifier_type + " - " + str(self.opt.n_synth_samples) + ":\n"
		output_text += str(acc_seen) + ", " + str(acc_unseen) + ", " + str(acc_h) + "\n\n"
		with open(join(self.opt.path, "gzsc_output.txt"), "a") as file:
			file.write(output_text)
	
	def run_zsr(self, k=10, use_gzsl_checkpoint=False):
		"""Zero-Shot Retrieval
		"""
		print("\nZero-Shot Retrieval")
		self._load_checkpoint(use_gzsl_checkpoint)
		n_samples = self.opt.n_synth_samples
		#syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		syn_features, syn_labels = generate_synthetic_features_mean(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		# combine all features, labels and their locations for finding matches
		all_features = self.data_handler.test_unseen_features
		all_labels = self.data_handler.test_unseen_labels
		all_loc = self.data_handler.test_unseen_loc
		# store matches for each class in out dict
		out = {}
		for i in range(syn_labels.size(0)):
			print("calculating distance for the centroid of class %d" %(syn_labels[i].item()))
			dists = torch.tensor([torch.dist(syn_features[i], all_features[j]) for j in range(len(all_features))]).float()
			#dists = torch.tensor([torch.sqrt(torch.sum((syn_features[i] - all_features[j])**2)) for j in range(len(all_features))]).float()
			out[syn_labels[i].item()] = torch.topk(dists, k, largest=False)
		# results for plot
		res = {}
		for x in out.keys():
			cls_name = self.data_handler.class_names[x]
			res[cls_name] = []
			# add labels of the top-k matches
			res[cls_name].append(all_labels[out[x][1]].numpy().tolist())
			# add paths of the corresponding images
			image_files = self.data_handler.image_files[all_loc[out[x][1].cpu().numpy()]]
			image_files = [x[0].split("/")[-2:] for x in image_files] 
			image_files = [join(self.opt.images_path, join(x[0], x[1])) for x in image_files]
			res[cls_name].append(image_files)
			# add image match vector
			res[cls_name].append((all_labels[out[x][1]] == x).numpy().tolist())
			# add accuracy
			acc = torch.sum(all_labels[out[x][1]] == x).item() / float(k)
			res[cls_name].append(acc*100)
		mean_acc = float(sum([res[k][3] for k in res.keys()]))/len(res)
		print("Mean ZSR acc: %f" %(mean_acc))	
		if self.opt.dataset != "AWA1":
			self._draw_zsr(res, k, "zsr.png")

	def run_gzsr(self, k=10, use_gzsl_checkpoint=False):
		"""Zero-Shot Retrieval
		"""
		print("\nZero-Shot Retrieval")
		self._load_checkpoint(use_gzsl_checkpoint)
		n_samples = self.opt.n_synth_samples
		#syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		syn_features, syn_labels = generate_synthetic_features_mean(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		# combine all features, labels and their locations for finding matches
		all_features = torch.cat([self.data_handler.train_features, self.data_handler.test_seen_features, self.data_handler.test_unseen_features], dim=0)
		all_labels = torch.cat([self.data_handler.train_labels, self.data_handler.test_seen_labels, self.data_handler.test_unseen_labels], dim=0)
		all_loc = np.concatenate((self.data_handler.trainval_loc, self.data_handler.test_seen_loc, self.data_handler.test_unseen_loc), axis=0)
		# store matches for each class in out dict
		out = {}
		for i in range(syn_labels.size(0)):
			print("calculating distance for the centroid of class %d" %(syn_labels[i].item()))
			dists = torch.tensor([torch.dist(syn_features[i], all_features[j]) for j in range(len(all_features))]).float()
			#dists = torch.tensor([torch.sqrt(torch.sum((syn_features[i] - all_features[j])**2)) for j in range(len(all_features))]).float()
			out[syn_labels[i].item()] = torch.topk(dists, k, largest=False)
		# results for plot
		res = {}
		for x in out.keys():
			cls_name = self.data_handler.class_names[x]
			res[cls_name] = []
			# add labels of the top-k matches
			res[cls_name].append(all_labels[out[x][1]].numpy().tolist())
			# add paths of the corresponding images
			image_files = self.data_handler.image_files[all_loc[out[x][1].cpu().numpy()]]
			image_files = [x[0].split("/")[-2:] for x in image_files] 
			image_files = [join(self.opt.images_path, join(x[0], x[1])) for x in image_files]
			res[cls_name].append(image_files)
			# add image match vector
			res[cls_name].append((all_labels[out[x][1]] == x).numpy().tolist())
			# add accuracy
			acc = torch.sum(all_labels[out[x][1]] == x).item() / float(k)
			res[cls_name].append(acc*100)
		mean_acc = float(sum([res[k][3] for k in res.keys()]))/len(res)
		print("Mean GZSR acc: %f" %(mean_acc))	
		if self.opt.dataset != "AWA1":
			self._draw_zsr(res, k, "gzsr.png")

	def _draw_zsr(self, res, k, file_name):
		# save the dict into a file
		with open(self.opt.path + "/zsr.txt", "w") as file:
			file.write(json.dumps(res))
		cls_candidates = {}
		for c in res:
			imgs = listdir(join(self.opt.images_path, c))
			r = random.randint(0, len(imgs)-1)
			cls_candidates[c] = join(join(self.opt.images_path, c), imgs[r])

		# plot the results
		self.plotter.plot_zsr(res, cls_candidates, k, file_name)

	def draw_synth_features(self, use_gzsl_checkpoint, n_samples):
		"""Draws synthesized and ds features
		"""
		self._load_checkpoint(use_gzsl_checkpoint)
		name = "tsne_synth_features_"
		if use_gzsl_checkpoint:
			name += "gzsl.html"
		else:	
			name += "zsl.html"
		syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
		X = syn_features.cpu().numpy()
		Y = syn_labels.cpu().numpy()
		test_cls = self.data_handler.unseen_classes.cpu().numpy()

		print("Visualizing t-SNE results")
		self.plotter.plot_tsne_lbl(X, lbls=Y, tst_cls=test_cls, cls_names=self.data_handler.class_names, n_iter=1000, perplexity=55, name=name)
