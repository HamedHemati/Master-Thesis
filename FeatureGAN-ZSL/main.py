import argparse
from os.path import join
from trainers.zswgan_trainer import ZSWGANTrainer
from trainers.filmwgan_trainer import FiLMWGANTrainer
from trainers.wgan_trainer import WGANTrainer
from trainers.gan_trainer import GANTrainer
from trainers.aewgan_trainer import AEWGANTrainer


def run_trainer(opt):
    folder_name = f"{opt.name_index}_{opt.dataset}_{opt.model}_{opt.n_h_d}_" + \
                  f"{opt.n_h_g}_{opt.lr}_{opt.feat_type}_{opt.cls_emb_type}"

    opt.outputs_path = join(opt.outputs_path, folder_name)
    if opt.model == 'ZSWGAN' or opt.model == 'ZSWGAN2':
        trainer = ZSWGANTrainer(opt=opt)
        trainer.run()
    elif opt.model == 'FiLMWGAN' or opt.model == 'FiLMWGAN2':
        trainer = FiLMWGANTrainer(opt=opt)
        trainer.run()
    elif opt.model == 'WGAN':
        trainer = WGANTrainer(opt=opt)
        trainer.run()
    elif opt.model == 'GAN':
        trainer = GANTrainer(opt=opt)
        trainer.run() 
    elif opt.model == 'AEWGAN':
        trainer = AEWGANTrainer(opt=opt)
        trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--n-iter-d', type=int, default=5)
    parser.add_argument('--lambda-gp', type=float, default=10.0)
    parser.add_argument('--lambda-centl', type=float, default=1.0)
    parser.add_argument('--cls-weight', type=float, default=0.02)
    parser.add_argument('--n-cls-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n-z', type=int, default=85)
    parser.add_argument('--n-feat', type=int, default=2048)
    parser.add_argument('--n-cls-emb', type=int, default=85)
    parser.add_argument('--n-h-d', type=int, default=4096)
    parser.add_argument('--n-h-g', type=int, default=4096)
    parser.add_argument('--model', type=str, default='fCLS-WGAN')
    parser.add_argument('--use-cls', type=str, default='yes')
    parser.add_argument('--use-gp', type=str, default='yes')
    parser.add_argument('--use-cent-loss', type=str, default='yes')
    parser.add_argument('--dataset', type=str, default='AWA2')
    parser.add_argument('--feat-type', type=str, default='res101')
    parser.add_argument('--cls-emb-type', type=str, default='att')
    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--n-synth-samples', type=int, default=1000)
    parser.add_argument('--eval-zsl', type=str, default="yes")
    parser.add_argument('--eval-gzsl', type=str, default="yes")
    parser.add_argument('--use-valset', type=str, default='no')
    parser.add_argument('--random-seed', type=int, default=1010)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--outputs-path', type=str, default='results')
    parser.add_argument('--name-index', type=str, default='0')
    opt = parser.parse_args()

    run_trainer(opt)