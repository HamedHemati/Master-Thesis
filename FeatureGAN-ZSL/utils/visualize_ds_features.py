import argparse
import os
from .data_handler import DataHandler
from os.path import join
from plotter import Plotter
import torch 


def visualize_ds_features(opt, output_path):
    data_handler = DataHandler(opt)
    t = data_handler.train_features
    y = data_handler.train_labels
    
    feats = torch.zeros(1, len(data_handler.train_features[0]))
    lbls = []
    if opt.draw_trainset == 'yes':
	    # load train features
	    for lbl in torch.unique(data_handler.train_labels):
	        feats = torch.cat((feats, data_handler.train_features[data_handler.train_labels==lbl][1:opt.n_samples]))
	        lbls.extend([lbl]*opt.n_samples)
    if opt.draw_testset == 'yes':
	    # load test features
	    for lbl in torch.unique(data_handler.test_unseen_labels):
	        feats = torch.cat((feats, data_handler.test_unseen_features[data_handler.test_unseen_labels==lbl][1:opt.n_samples]))
	        lbls.extend([lbl]*opt.n_samples)
    feats = feats[1:]
    lbls = torch.tensor(lbls)
    cls_names = data_handler.all_classnames
    plotter = Plotter(output_path)
    plotter.plot_tsne_lbl(feats.numpy(), lbls=lbls.numpy(), tst_cls=set(lbls.numpy()), cls_names=cls_names,
                          n_iter=3000, perplexity=50, name=opt.dataset+'_'+opt.feat_type+'.html')


def visualize(opt):
    folder_name = opt.dataset  + '_' + opt.feat_type
    output_path = join(opt.output_path, folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    visualize_ds_features(opt, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-type', type=str, default='')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--cls-emb-type', type=str, default='att')
    parser.add_argument('--use-valset', type=str, default='no')
    parser.add_argument('--draw-trainset', type=str, default='yes')
    parser.add_argument('--draw-testset', type=str, default='yes')
    opt = parser.parse_args()
    visualize(opt)
