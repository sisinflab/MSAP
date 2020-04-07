import argparse
import os
import shutil

from dataset.dataset import DataLoader
from recommender.AMR import AMR
from recommender.BPRMF import BPRMF
from util.read import read_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run Attack.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='movielens-500',
                        help='dataset path: movielens-1m, gowalla, lastfm, yelp')
    parser.add_argument('--rec', nargs='?', default="amr", help="bprmf, amr")
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs (Not Used in Run Attack)')
    parser.add_argument('--verbose', type=int, default=500,
                        help='number of epochs to show the results ans store model parameters.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--reg', type=float, default=0, help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--restore_epochs', type=int, default=1,
                        help='Default is 1: It is the epoch value from which the attack will be executed.')

    # Parameters useful during the adv. training
    parser.add_argument('--adv_type', nargs='?', default="fgsm", help="fgsm, future work other techniques...")
    parser.add_argument('--adv_reg', type=float, default=0, help='Regularization for adversarial loss')
    parser.add_argument('--eps', type=float, default=0.5, help='Epsilon for adversarial weights.')

    # Parameters useful during the adv. attack
    parser.add_argument('--attack_type', nargs='?', default="fgsm", help="fgsm, bim, pgd, deepFool, ...")
    parser.add_argument('--attack_users', nargs='?', default="full", help="full, random (to be implemented), ...")
    parser.add_argument('--attack_eps', type=float, default=0.5, help='Epsilon for adversarial ATTACK.')
    parser.add_argument('--attack_step_size', type=int, default=4, help='Step Size for BIM/PGD ATTACK.')
    parser.add_argument('--attack_iteration', type=int, default=10, help='Iterations for BIM/PGD ATTACK.')

    return parser.parse_args()


def attack():
    args = parse_args()
    args.restore_epochs = args.epochs
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

    data = DataLoader(path_train_data=path_train_data
                      , path_test_data=path_test_data)

    print("RUNNING {0} Attack on DATASET {1} and Recommender {2}".format(args.attack_type, args.dataset, args.rec))
    print("- PARAMETERS:")
    for arg in vars(args):
        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
    print("\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Initialize the model under attack
    if args.rec == 'bprmf':
        model = BPRMF(data, path_output_rec_result, path_output_rec_weight, args)
    elif args.rec == 'amr':
        model = AMR(data, path_output_rec_result, path_output_rec_weight, args)
    else:
        raise NotImplementedError('Unknown Recommender Model.')

    # Restore the Model Parameters
    if not model.restore():
        raise NotImplementedError('Unknown Restore Point/Model.')

    # Initialize the Attack
    if args.attack_users == 'full':
        # Start full batch attacks
        if args.attack_type == 'fgsm':
            attack_name = '{0}_ep{1}_sz{2}_'.format(args.attack_type, args.attack_eps, args.attack_users)
            model.attack_full_fgsm(args.attack_eps, attack_name)
        elif args.attack_type in ['bim', 'pgd']:
            attack_name = '{0}{1}_ep{2}_es{3}_sz{4}_'.format(args.attack_type, args.attack_iteration, args.attack_eps, args.attack_step_size,
                                                          args.attack_users)
            model.attack_full_iterative(args.attack_type, args.attack_iteration, args.attack_eps, args.attack_step_size, attack_name)



    else:
        raise NotImplementedError('Unknown Attack USERS STRATEGY.')


if __name__ == '__main__':
    attack()
