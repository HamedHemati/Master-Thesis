import argparse
from utils.tester import Tester


def run_tester(opt):
    compute_confusion = False
    if opt.compute_confusion == "yes":
        compute_confusion = True
    
    tester = Tester(opt)
    if opt.zsc == "yes":
        tester.run_zsl(compute_confusion=compute_confusion)
    if opt.gzsc== "yes":
        tester.run_gzsl(compute_confusion=compute_confusion)
    if opt.tsne_synth == "yes":
        tester.draw_synth_features(use_gzsl_checkpoint=True, n_samples=50)
        tester.draw_synth_features(use_gzsl_checkpoint=False, n_samples=50)
    if opt.zsr == "yes":
        tester.run_zsr(k=5, use_gzsl_checkpoint=False)
    if opt.gzsr == "yes":
        tester.run_gzsr(k=5, use_gzsl_checkpoint=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-z', type=int, default=85)
    parser.add_argument('--n-feat', type=int, default=2048)
    parser.add_argument('--n-cls-emb', type=int, default=85)
    parser.add_argument('--n-h-d', type=int, default=4096)
    parser.add_argument('--n-h-g', type=int, default=4096)
    parser.add_argument('--model', type=str, default='fCLS-WGAN')
    parser.add_argument('--dataset', type=str, default='AWA2')
    parser.add_argument('--use-valset', type=str, default='no')
    parser.add_argument('--feat-type', type=str, default='res101')
    parser.add_argument('--cls-emb-type', type=str, default='att')
    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--images-path', type=str, default='')
    parser.add_argument('--n-synth-samples', type=int, default=1000)
    parser.add_argument('--eval-zsl', type=str, default="yes")
    parser.add_argument('--eval-gzsl', type=str, default="yes")
    parser.add_argument('--random-seed', type=int, default=1010)
    parser.add_argument('--outputs-path', type=str, default='results')
    parser.add_argument('--epoch', type=str, default='0')
    parser.add_argument('--name-index', type=int, default=0)
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--classifier-type', type=str, default='linsoftmax')
    parser.add_argument('--compute-confusion', type=str, default='no')
    parser.add_argument('--tsne-synth', type=str, default='no')
    parser.add_argument('--zsc', type=str, default='no')
    parser.add_argument('--gzsc', type=str, default='no')
    parser.add_argument('--zsr', type=str, default='no')
    parser.add_argument('--gzsr', type=str, default='no')
    opt = parser.parse_args()
    run_tester(opt)
