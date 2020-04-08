import argparse
import os
import shutil

from dataset.dataset import DataLoader
from recommender.AMR import AMR
from recommender.BPRMF import BPRMF
from util.read import read_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run train of the Recommender Model.")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dataset', nargs='?', default='movielens-1m', help='dataset path: movielens-1m, gowalla, lastfm, yelp')
    parser.add_argument('--rec', nargs='?', default="amr", help="bprmf, amr")
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=1000, help='number of epochs to show the results ans store model parameters.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--reg', type=float, default=0, help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--restore_epochs', type=int, default=1000, help='Default is 1: The restore epochs (Must be lower than the epochs)')
    parser.add_argument('--eps', type=float, default=0.5, help='Epsilon for adversarial weights.')
    parser.add_argument('--adv_type', nargs='?', default="fgsm", help="fgsm, future work other techniques...")
    parser.add_argument('--adv_reg', type=float, default=1, help='Regularization for adversarial loss')

    return parser.parse_args()


def manage_directories(path_output_rec_result, path_output_rec_weight):
    if os.path.exists(os.path.dirname(path_output_rec_result)):
        shutil.rmtree(os.path.dirname(path_output_rec_result))
    os.makedirs(os.path.dirname(path_output_rec_result))
    if os.path.exists(os.path.dirname(path_output_rec_weight)):
        shutil.rmtree(os.path.dirname(path_output_rec_weight))
    os.makedirs(os.path.dirname(path_output_rec_weight))


def train():
    args = parse_args()
    path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight = read_config(
        sections_fields=[('PATHS', 'InputTrainFile'),
                         ('PATHS', 'InputTestFile'),
                         ('PATHS', 'OutputRecResult'),
                         ('PATHS', 'OutputRecWeight')])
    path_train_data, path_test_data, = path_train_data.format(
        args.dataset), path_test_data.format(args.dataset)

    if args.rec == 'bprmf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')
    elif args.rec == 'amr':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.eps),
                                                               '' + args.adv_type)

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.eps),
                                                               '' + args.adv_type)

    # Create directories to Store Results and Rec Models
    manage_directories(path_output_rec_result, path_output_rec_weight)

    data = DataLoader(path_train_data=path_train_data
                      , path_test_data=path_test_data)

    print("RUNNING {0} Training on DATASET {1}".format(args.rec, args.dataset))
    print("- PARAMETERS:")
    for arg in vars(args):
        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
    print("\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.rec == 'bprmf':
        model = BPRMF(data, path_output_rec_result, path_output_rec_weight, args)
    elif args.rec == 'amr':
        model = AMR(data, path_output_rec_result, path_output_rec_weight, args)
    else:
        raise NotImplementedError('Unknown Recommender Model.')
    model.train()


if __name__ == '__main__':
    train()
