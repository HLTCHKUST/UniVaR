import multiprocessing
import argparse
import os
import torch
from torch.utils.data import DataLoader
# tensorboard
from torch.utils.tensorboard import SummaryWriter

# from sentence_transformers import SentenceTransformer
from customed_sentence_transformers import CustomedSentenceTransformer as SentenceTransformer # add teensorboard writer

from loss import SiameseContrastiveTensionLossInBatchNegatives
from utils import get_params_count_summary, init_random_seed, expand_sentence_transformer_encoder
from data import save_analysis_data
from evaluation.validation_loss import LossEvaluator

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model args
    parser.add_argument('--train_dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the dataset')    
    parser.add_argument('--out_dataset_path', type=str, help='Path to the output dataset')
    parser.add_argument('--model_path', type=str, default='distiluse-base-multilingual-cased', help='Path to the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--per_value_sample_truncation_train', type=int, default=-1, help='Truncate the sample to this length')
    parser.add_argument('--per_value_sample_truncation_val', type=int, default=-1, help='Truncate the sample to this length')
    parser.add_argument('--per_value_sample_truncation_test', type=int, default=-1, help='Truncate the sample to this length')
    parser.add_argument('--val_split_ratio', type=float, default=0.005, help='Validation split, set to zero to disable validation during training')
    parser.add_argument('--include_no_value', default=False, action='store_true', help='Whether to include no value data')

    # Random Seed
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--comment', type=str, default='value-embedding', help='Name of the run')

    # Training hyperparameters
    parser.add_argument('--scheduler', type=str, default="WarmupLinear", help='Scheduler')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Optimizer params')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--evaluation_steps', type=int, default=0, help='Evaluation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='Max grad norm')
    parser.add_argument('--checkpoint_save_steps', type=int, default=500, help='Checkpoint save steps')
    
    # Extend Model
    parser.add_argument('--expand_encoder', default=False, action='store_true', help='Whether to expand the encoder or not')
    parser.add_argument('--num_new_layer', type=int, default=12, help='Number of new encoder layers')
    parser.add_argument('--retain_prior_modules', default=False, action='store_true', help='Whether to freeze the prior encoder module before the new one')

    args = parser.parse_args()
    init_random_seed(args.random_seed)
        
    # Init the model
    model = SentenceTransformer(args.model_path, writer=None)
    if args.expand_encoder:
        model = expand_sentence_transformer_encoder(
            model, num_extended_layers=args.num_new_layer, freeze_prior_modules=args.retain_prior_modules
        )
    model = model.to('cuda')
    train_loss = torch.compile(SiameseContrastiveTensionLossInBatchNegatives(model=model)).to('cuda')

    # Instantiate Datalaoder
    save_analysis_data(
        train_dataset_path=args.train_dataset_path, 
        test_dataset_path=args.test_dataset_path,
        val_split_ratio=args.val_split_ratio,
        include_no_value=args.include_no_value,
        outpath=args.out_dataset_path
    )