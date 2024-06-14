import multiprocessing
import argparse
import os
import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from training_utils.customed_sentence_transformers import CustomedSentenceTransformer as SentenceTransformer
from training_utils.customed_evaluator_during_training import ValueEvaluator
from training_utils.utils import get_params_count_summary, init_random_seed, expand_sentence_transformer_encoder

from models.infonce import InfoNCE, ValueAwareInfoNCE
from models.barlowtwins import BarlowTwins

from sentence_transformers.models.Normalize import Normalize
from training_utils.customed_sentence_transformers import CustomedSentenceTransformer as SentenceTransformer # add teensorboard writer

from training_utils.utils import get_params_count_summary, init_random_seed, expand_sentence_transformer_encoder

from data import get_data

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--train_dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--per_value_train_samples', type=int, default=-1, help='Truncate the sample to this length')
    parser.add_argument('--per_value_val_samples', type=int, default=-1, help='Truncate the sample to this length')
    parser.add_argument('--per_value_test_samples', type=int, default=-1, help='Truncate the sample to this length')
    parser.add_argument('--include_no_value', default=False, action='store_true', help='Whether to include no value data')

    # add min_qa_per_view=1, max_qa_per_view=1
    parser.add_argument('--qa_per_view', type=int, default=1, help='Minimum number of QA pairs per view')
    
    # Model and training args
    parser.add_argument('--model_path', type=str, default='distiluse-base-multilingual-cased', help='Path to the model')
    parser.add_argument('--loss', type=str, default='infonce', help='value bert loss name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--skip_torch_compile', default=False, action='store_true', help='Whether to use torch compile or not')

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
    writer = SummaryWriter(comment=f"_{args.comment}")
        
    # Init the model
    model = SentenceTransformer(args.model_path, writer=writer)
    if type(model[-1]) is Normalize:
        del model[-1] # Remove the last normalization Layer
        
    if args.expand_encoder:
        model = expand_sentence_transformer_encoder(
            model, num_extended_layers=args.num_new_layer, freeze_prior_modules=args.retain_prior_modules
        )
    model = model.to('cuda')

    if args.include_no_value:
        train_loss = ValueAwareInfoNCE(model=model)
    elif args.loss == 'infonce':
        train_loss = InfoNCE(model)
    elif args.loss == 'barlowtwins':
        train_loss = BarlowTwins(model)
    else:
        raise ValueError(f"Loss {args.loss} not found")

    if not args.skip_torch_compile:
        train_loss = torch.compile(train_loss).to('cuda')
    else:
        train_loss = train_loss.to('cuda')

    print(get_params_count_summary(model))
    print(train_loss)
    for k, v in vars(args).items():
        writer.add_text(k, str(v))
        print(f"\t{k}: {v}")

    # Instantiate Datalaoder
    train_dataset, validation_dataset = get_data(
        args.train_dataset_path, 
        args.per_value_train_samples,
        args.per_value_val_samples,
        args
        )
    test_dataset, _ = get_data(
        args.test_dataset_path, 
        args.per_value_test_samples,
        -1,
        args
        )
    
    assert len(train_dataset) > args.train_batch_size
    assert len(validation_dataset) > args.eval_batch_size
    assert len(test_dataset) > args.eval_batch_size
    
    train_loader = DataLoader(
        train_dataset, shuffle=True, 
        num_workers=multiprocessing.cpu_count() if args.train_batch_size > multiprocessing.cpu_count() else args.train_batch_size, 
        batch_size=args.train_batch_size,  collate_fn=lambda x: x, drop_last=True, pin_memory=True
    )

    seen_llms_unseen_qa_loader = DataLoader(
        validation_dataset, 
        num_workers=multiprocessing.cpu_count() if args.eval_batch_size > multiprocessing.cpu_count() else args.eval_batch_size, 
        batch_size=args.eval_batch_size, collate_fn=lambda x: x, drop_last=True
    )

    unseen_llms_loader = DataLoader(
        test_dataset, 
        num_workers=multiprocessing.cpu_count() if args.eval_batch_size > multiprocessing.cpu_count() else args.eval_batch_size, 
        batch_size=args.eval_batch_size, collate_fn=lambda x: x, drop_last=True
    )

    evaluator = ValueEvaluator(
        loaders={
            'seen_llms_unseen_qa': seen_llms_unseen_qa_loader,
            'unseen_llms': unseen_llms_loader
        }, loss_model=train_loss, 
        name='validation', logs_writer=writer,
        show_progress_bar=True
    )

    # Train & Validation Phase
    model.fit(
        [(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        optimizer_params={"lr": args.lr},
        weight_decay=args.weight_decay,
        evaluation_steps=args.evaluation_steps,
        output_path=writer.log_dir,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=writer.log_dir,
        use_amp=True,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=10,
        eval_before_start=True,
    )
