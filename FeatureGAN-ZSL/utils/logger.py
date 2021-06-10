import os
import shutil


class Logger(object):
    def __init__(self, save_dir, opt):
        self.save_dir = save_dir
        self.opt = opt
        self.log_filename = 'params.txt'
        self._init_directories()
        self._init_logfile()

    def _init_directories(self):
        # create/initialize folders
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        os.makedirs(os.path.join(self.save_dir, 'net_d_checkpoints'))
        os.makedirs(os.path.join(self.save_dir, 'net_g_checkpoints'))
        os.makedirs(os.path.join(self.save_dir, 'hists'))

    def _init_logfile(self):
        # create a log file and initialize it with training parameters info
        log_msg = ''
        for key_name in self.opt.__dict__:
            log_msg += key_name + ':' + str(self.opt.__dict__[key_name]) + '\n'
        print(log_msg + '\n')
        with open(os.path.join(self.save_dir, self.log_filename), 'w') as log_file:
            log_file.write(log_msg)

    def create_model_log(self, net_d, net_g):
        log_msg = "NetD:\n" + str(net_d) + "\n\nNetG:\n" + str(net_g)
        with open(os.path.join(self.save_dir, "model.txt"), 'w') as log_file:
            log_file.write(log_msg)