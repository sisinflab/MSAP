import argparse
import os
import shutil

from src.dataset.dataset import DataLoader
from src.recommender.BPRMF import BPRMF
from src.util.read import read_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run train of the Recommender Model.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rec', nargs='?', default="bprmf", help="bprmf, amr")
    parser.add_argument('--dataset', nargs='?', default='movielens-500',
                        help='dataset path: movielens-1m, gowalla, lastfm, yelp')
    parser.add_argument('--verbose', type=int, default=10, help='number of epochs to show the results.')
    parser.add_argument('--k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--reg', type=float, default=0, help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--reg_adv', type=float, default=1, help='Regularization for adversarial loss')
    parser.add_argument('--restore', type=str, default=None, help='The restore time_stamp for weights in \Pretrain')
    parser.add_argument('--ckpt', type=int, default=250, help='Save the model per X epochs.')
    parser.add_argument('--adv_epoch', type=int, default=3000,
                        help='Epoch of bpr-mf pre-trained to apply adversarial regularization.')
    parser.add_argument('--eps', type=float, default=0, help='Epsilon for adversarial weights.')
    parser.add_argument('--adv', type=int, default=0, help='0 - No adversarial, 1 adversarial training')

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

    path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                           args.rec,
                                                           args.embed_size,
                                                           args.epochs,
                                                           0)

    path_output_rec_weight = path_output_rec_result.format(args.dataset,
                                                           args.rec,
                                                           args.embed_size,
                                                           args.epochs,
                                                           0)

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

    model.train()


if __name__ == '__main__':
    train()
