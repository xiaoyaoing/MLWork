import argparse




def base_parser():
    parser = argparse.ArgumentParser(description='KNN-Softmax Training')

    # hype-parameters
    parser.add_argument('-lr', type=float, default=1e-4,
                        help="learning rate of new parameters")
    parser.add_argument('-tradeoff', type=float, default=1.0,
                        help="learning rate of new parameters")
    parser.add_argument('-exp', type=str, default='exp1',
                        help="learning rate of new parameters")
    parser.add_argument('-margin', type=float, default=0.0,
                        help="margin for metric loss")

    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('-num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('-dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('-alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in KNN Softmax')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')

    # network
    parser.add_argument('-data', default='cifar100', required=False,
                        help='path to Data Set')
    parser.add_argument('-net', default='resnet18')
    parser.add_argument('-loss', default='triplet_no_hard_mining', required=False,
                        help='loss for training network')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='epochs for training process')

    parser.add_argument('-seed', default=1993, type=int, metavar='N',
                        help='seeds for training process')
    parser.add_argument('-save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')
    parser.add_argument('-lr_step', default=200, type=int, metavar='N',
                        help='number of epochs to save model')
    # Resume from checkpoint
    parser.add_argument('-start', default=0, type=int,
                        help='resume epoch')

    # basic parameter
    parser.add_argument('-log_dir', default='cifar100',
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument("-gpu", type=str, default='0',
                        help='which gpu to choose')
    parser.add_argument("-method", type=str,
                        default='Finetuning', help='Choose FT or SC')

    parser.add_argument('-mapping_mean', default='no',
                        type=str, help='mapping')
    parser.add_argument('-sigma', default=0.0, type=float, help='sigma')
    parser.add_argument('-vez', default=0, type=int, help='vez')
    parser.add_argument('-task', default=11, type=int, help='vez')
    parser.add_argument('-base', default=50, type=int, help='vez')

    args = parser.parse_args()
    return args
