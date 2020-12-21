"""Arguments definition file, used to save and modify all the arguments in the project.
Author:
    Chris Chute (chute@stanford.edu)
Edior:
    Jiarui Feng
"""

import argparse
PROJECT_NAME="SigGraInferNet"



def get_train_args():
    """Get arguments needed in training the model."""
    parser = argparse.ArgumentParser(f'Train a model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)
    add_train_args(parser)
    #Add other arguments if needed

    args = parser.parse_args()

    if args.metric_name == 'Loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ("Accuracy","Recall","AUC"):
        # Best checkpoint is the one that maximizes Accuracy or recall
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args



def get_test_args():
    """Get arguments needed in testing the model."""
    parser = argparse.ArgumentParser(f'Test a trained model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)

    # Require mdoel load_path for testing
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args


def add_common_args(parser):
    """Add arguments common in all files, typically file directory"""

    parser.add_argument('--data_dir',
                    type=str,
                    default='../data')

    parser.add_argument('--train_file',
                    type=str,
                    default='data/equal_train_data_all.npz')

    parser.add_argument('--dev_file',
                    type=str,
                    default='data/equal_test_data_all.npz')

    parser.add_argument('--test_file',
                    type=str,
                    default='data/equal_test_data_all.npz')


    parser.add_argument('--PPI_dir',
                        type=str,
                        default='data/STRING_all.json')

    parser.add_argument('--PPI_gene_query_dict_dir',
                        type=str,
                        default="data/string_gene_query_dict_all.json")

    parser.add_argument('--PPI_gene_feature_dir',
                        type=str,
                        default="data/STRING_gene_feature_all.npz")




def add_train_test_args(parser):
    """Add arguments common to training and testing"""
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')


    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')

    parser.add_argument('--feature_input_size',
                        type=int,
                        default=130,
                        help='Number of features in gene input.')

    parser.add_argument('--feature_output_size',
                        type=int,
                        default=32,
                        help='Number of features in gene representation.')

    parser.add_argument('--PPI_input_size',
                        type=int,
                        default=50,
                        help='Number of features in protein-protein interaction data input.')

    parser.add_argument('--PPI_output_size',
                        type=int,
                        default=32,
                        help='Number of features in protein-protein interaction data output.')

    parser.add_argument('--num_GNN',
                        type=int,
                        default=3,
                        help='Number of GNN layer.')

    parser.add_argument('--num_head',
                        type=int,
                        default=4,
                        help='Number of head in GAT.')

    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

    parser.add_argument('--max_nodes',
                        type=int,
                        default=200,
                        help='Maximum number of nodes in subgraph.')

    parser.add_argument('--model_name',
                        type=str,
                        default="GAT",
                        help='Which model to train or test')
    parser.add_argument('--gamma',
                        type=float,
                        default=1.,
                        help='gamma of focal loss.')
    parser.add_argument('--alpha',
                        type=float,
                        default=1.,
                        help='alpha of focal loss.(first dimension)')


def add_train_args(parser):
    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=3e-7,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='Accuracy',
                        choices=('Accuracy', 'Recall', 'Loss','AUC'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

