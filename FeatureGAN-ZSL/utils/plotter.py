import plotly.offline as py
import plotly.graph_objs as go
from os.path import join
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image, ImageDraw


class Plotter(object):
    def __init__(self, path):
        self.path = path

    def plot_classification_loss(self, x=[], losses=[]):
        """Plots epoch/loss graph of a classification task.
        """
        pl = go.Scatter(x=x, y=losses, mode='lines', name='Loss')
        data = [pl]
        layout = go.Layout(
            title='Feature Classification',
            xaxis=dict(
                title='Epoch',
            ),
            yaxis=dict(
                title='Classification Loss',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=join(self.path, 'cls_loss.html'), auto_open=False)

    def plot_gan_train(self, plot_hists, ws_distance):
        """Plots GAN graphs.
        """
        # plot losses
        x = plot_hists['x']
        pl1 = go.Scatter(x=x, y=plot_hists['loss_net_d'], mode='lines', name="Loss - Discriminator")
        pl2 = go.Scatter(x=x, y=plot_hists['loss_net_g'], mode='lines', name="Loss - Generator")
        data = [pl1, pl2]
        layout = go.Layout(
            title='',
            xaxis=dict(
                title='Epoch',
            ),
            yaxis=dict(
                title='Loss',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=join(self.path, 'train_loss.html'), auto_open=False)

        # plot means
        pl4 = go.Scatter(x=x, y=plot_hists['mean_d_x'], mode='lines', name='Mean D(x)')
        pl5 = go.Scatter(x=x, y=plot_hists['mean_d_z1'], mode='lines', name='Mean D(G(z))')
        pl6 = go.Scatter(x=x, y=plot_hists['mean_d_z2'], mode='lines', name='Mean D*(G(z))')
        data = [pl4, pl5, pl6]
        layout = go.Layout(
            title='',
            xaxis=dict(
                title='Epoch',
            ),
            yaxis=dict(
                title='Mean',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=join(self.path, 'train_means.html'), auto_open=False)

        if ws_distance:
            # plot wasserstein distance
            pl1 = go.Scatter(x=x, y=plot_hists['loss_net_d'], mode='lines', name="Wasserstein Distance")
            data = [pl1]
            layout = go.Layout(
                title='',
                xaxis=dict(
                    title='Epoch',
                ),
                yaxis=dict(
                    title='WS Distance',
                )
            )
            fig = go.Figure(data=data, layout=layout)
            py.plot(fig, filename=join(self.path, 'train_ws_dist.html'), auto_open=False)

    def plot_gan_eval(self, plot_hists, gzsl):
        if not gzsl:
            x = list(range(len(plot_hists['zsl'])))
            pl_zsl = go.Scatter(x=x, y=plot_hists['zsl'], mode='lines', name='ZSL')
            data = [pl_zsl,]
            layout = go.Layout(
                title='GZSL Accuracy',
                xaxis=dict(
                    title='Epoch',
                ),
                yaxis=dict(
                    title='Accuracy',
                )
            )
            fig = go.Figure(data=data, layout=layout)
            py.plot(fig, filename=join(self.path, 'train_eval_zsl.html'), auto_open=False)

        if gzsl:
            x = list(range(len(plot_hists['gzsl_seen'])))
            pl_gzsl_seen = go.Scatter(x=x, y=plot_hists['gzsl_seen'], mode='lines', name='GZSL - Seen')
            pl_gzsl_unseen = go.Scatter(x=x, y=plot_hists['gzsl_unseen'], mode='lines', name='GZSL - Unseen')
            pl_gzsl_hmean = go.Scatter(x=x, y=plot_hists['gzsl_hmean'], mode='lines', name='GZSL - H-Mean')
            data = [pl_gzsl_seen, pl_gzsl_unseen, pl_gzsl_hmean,]
            layout = go.Layout(
                title='GZSL Accuracy',
                xaxis=dict(
                    title='Epoch',
                ),
                yaxis=dict(
                    title='Accuracy',
                )
            )
            fig = go.Figure(data=data, layout=layout)
            py.plot(fig, filename=join(self.path, 'train_eval_gzsl.html'), auto_open=False)

    def plot_accuracy(self, plot_hists, file_name):
        x = list(range(len(plot_hists['acc_zsl'])))
        pl_zsl = go.Scatter(x=x, y=plot_hists['acc_zsl'], mode='lines', name='ZSL')
        pl_gzsl_seen = go.Scatter(x=x, y=plot_hists['acc_gzsl_seen'], mode='lines', name='GZSL - Seen')
        pl_gzsl_unseen = go.Scatter(x=x, y=plot_hists['acc_gzsl_unseen'], mode='lines', name='GZSL - Unseen')
        pl_gzsl_hmean = go.Scatter(x=x, y=plot_hists['acc_gzsl_hmean'], mode='lines', name='GZSL - H-Mean')
        data = [pl_zsl, pl_gzsl_seen, pl_gzsl_unseen, pl_gzsl_hmean,]
        layout = go.Layout(
            title='ZSL and GZSL Accuracy Graph',
            xaxis=dict(
                title='Epoch',
            ),
            yaxis=dict(
                title='Accuracy',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=join(self.path, file_name), auto_open=False)

    def plot_accuracy_single(self, plot_hists, file_name):
        x = list(range(len(plot_hists['acc_zsl'])))
        pl_zsl = go.Scatter(x=x, y=plot_hists['acc_zsl'], mode='lines', name='ZSL')
        data = [pl_zsl,]
        layout = go.Layout(
            title='ZSL Accuracy Graph',
            xaxis=dict(
                title='Epoch',
            ),
            yaxis=dict(
                title='Accuracy',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=join(self.path, file_name), auto_open=False)
            
    def plot_tsne_lbl(self, X, lbls, tst_cls, cls_names, n_components=2, verbose=1, perplexity=60,
                      n_iter=500, name='tsne.html'):
        """Plots the t-SNE visualization of a set of features with given labels.
        """
        tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
        results = tsne.fit_transform(X)
        plots = []
        for c in tst_cls:
            res = [results[i] for i in range(len(results)) if lbls[i] == c]
            res = np.array(res)
            if len(res) != 0:
                pl = go.Scatter(x=list(res[:, 0]), y=list(res[:, 1]), mode='markers',
                                marker=dict(size=5, color=c, colorscale='Jet', opacity=0.8), name=cls_names[c])
                plots.append(pl)
        py.plot(plots, filename=join(self.path, name), auto_open=False)

    def plot_confusion_matrix(self, x, y, z, file_name='confusion-matrix.html'):
        """Plots confusion matrix for classification results.
        """
        trace1 = {"x": x, "y": y, "z": z,
                  "colorscale": "YIGnBu", "type": "heatmap"}
        data = go.Data([trace1])
        layout = {
            "barmode": "overlay",
            "title": "Confusion Matrix",
            "xaxis": {"title": "",
                      "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 15}},
            "yaxis": {"autorange": "reversed", "title": "", "titlefont": {
                      "color": "#7f7f7f", "family": "Courier New, monospace", "size": 15}}
            }
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=join(self.path, file_name), auto_open=False)

    def plot_zsr(self, res, cls_candidates, k, file_name):
        """Plots topk images in zero-shot retrieval for each test class
        """
        # select class candidates
        tile_size = 100
        space = 20
        left_margin = tile_size + int(tile_size/2)
        image_width = (k * (tile_size + space)) + space + left_margin
        image_height = (len(res) * (tile_size + space)) + space
        main_image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        draw = ImageDraw.Draw(main_image)
        i = 0

        for c in res:
            offset_h = (i * tile_size) + (i+1) * space
            # draw class candidate
            img = Image.open(cls_candidates[c], 'r').resize((tile_size, tile_size))
            main_image.paste(img, (space, offset_h))
            j = 0
            for img_file in res[c][1]:
                offset_w = left_margin + (j * tile_size) + (j+1) * space
                if res[c][2][j] == 0:
                    color = "red"
                else:
                    color = "green"    
                top_left = (offset_w-int(space/2)+2, offset_h-int(space/2)+2)
                below_right = (offset_w + tile_size + int(space/2)-2, offset_h+tile_size+int(space/2)-2)
                draw.rectangle((top_left, below_right), fill=color)
                img = Image.open(img_file, 'r').resize((tile_size, tile_size))
                main_image.paste(img, (offset_w, offset_h))
                j += 1
            i += 1
        main_image.save(join(self.path, file_name))    