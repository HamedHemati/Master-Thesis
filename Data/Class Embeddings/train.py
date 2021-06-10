import os 
import torch
import torch.nn as nn
from torch import optim
from argparse import ArgumentParser
from models import LSTM, GRU
from data_handler import DataHandler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(model, hidden_size, n_epochs, lr, output_path, data_handler):
	cuda = torch.cuda.is_available()
	data_loader = data_handler.get_dataloader()
	num_cls = data_handler.num_cls
	fastText_emb_size = 300
	if model == 'lstm':
		rnn = LSTM(fastText_emb_size, hidden_size, num_cls)
		print('model: lstm')
	elif model == 'gru':
		rnn = GRU(fastText_emb_size, hidden_size, num_cls)
		print('model: gru')
	criterion = nn.NLLLoss()

	if cuda:
		rnn.cuda()
		criterion.cuda()

	optimizer = optim.Adam(rnn.parameters(), lr=lr) #, weight_decay=1e-4)
	#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
	log_txt = ""
	epoch_losses = []
	hidden0 = torch.zeros(1, 1, hidden_size)
	if cuda:
		hidden0 = hidden0.cuda()
	for epoch in range(n_epochs):
		sum_epoch_loss = 0.0
		for itr,(sent, cls) in enumerate(data_loader):
			#lr_scheduler.step()
			rnn.zero_grad()
			if cuda:
				sent, cls = sent.cuda(), cls.cuda()
			out_cls, _ = rnn(sent[0].unsqueeze(1), hidden0)
			loss = criterion(out_cls.view(1, num_cls), cls[0])
			loss.backward()
			optimizer.step()
			sum_epoch_loss += loss.item()
			if itr % 10 ==0:
				msg = "epoch %d - %.2f %%, loss: %f" % (epoch, (float(itr)/len(data_loader))*100, loss.item())
				print(msg)
				log_txt += msg + "\n"
		avg_epoch_loss = float(sum_epoch_loss) / len(data_loader)

		# save checkpoint
		if epoch > 0:
			if avg_epoch_loss < min(epoch_losses):
				torch.save(rnn.cpu().state_dict(), output_path + '/checkpoint.pth')
				print("checkpoint is saved")
				with open(output_path + "/log.txt", "w") as file:
					file.write(log_txt)
				if cuda:
					rnn.cuda()

		epoch_losses.append(avg_epoch_loss)
		plt.plot(epoch_losses)
		plt.savefig(output_path + '/loss_graph.png')

def main(opt):
	output_path = 'outputs/' + opt.dataset + '_' + opt.model + '_hidden-size' + str(opt.hidden_size) + '_nitr' + str(opt.n_epochs) + '_lr' + str(opt.lr)
	if not os.path.exists(output_path):
		os.makedirs(output_path)	
	data_handler = DataHandler(opt)
	train(opt.model, opt.hidden_size, opt.n_epochs, opt.lr, output_path, data_handler)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fastText-model-path', type=str, default="")
    parser.add_argument('--dataset', type=str, default="awa2")
    parser.add_argument('--model', type=str, default="lstm")
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--n-epochs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=0.01)
    opt = parser.parse_args()
    main(opt)
