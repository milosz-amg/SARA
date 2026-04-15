"""
Fine-tune BGE-base-en-v1.5 for ArXiv Paper Similarity

Supports two loss functions:
  --loss cosent : CoSENTLoss with continuous similarity scores (hierarchical + fuzzy)
  --loss mnr    : MultipleNegativesRankingLoss with in-batch negatives (legacy)

Default is CoSENTLoss which uses scored pairs from 05_prepare_finetune_data.py --mode scored.
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATA_DIR, PROJECT_ROOT

from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CoSENTLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FINETUNE_DIR = os.path.join(DATA_DIR, 'finetune')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BGE-base for ArXiv paper similarity')
    parser.add_argument('--model', type=str, default='BAAI/bge-base-en-v1.5',
                        help='Base model to fine-tune (default: BAAI/bge-base-en-v1.5)')
    parser.add_argument('--loss', type=str, default='cosent', choices=['cosent', 'mnr'],
                        help='Loss function: cosent (scored pairs) or mnr (anchor/positive pairs)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for regularization (default: 0.01)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio (default: 0.1)')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Evaluation interval in steps (default: 100)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Fine-tune data directory (default: data/finetune/)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for model (default: models/bge-base-arxiv-finetuned)')

    args = parser.parse_args()

    data_dir = args.data_dir or FINETUNE_DIR
    output_dir = args.output_dir or os.path.join(MODELS_DIR, 'bge-base-arxiv-finetuned')

    logger.info("=" * 60)
    logger.info("Fine-tuning BGE for ArXiv Paper Similarity")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.model}")
    logger.info(f"Loss: {args.loss}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Weight decay: {args.weight_decay}")

    # Detect dataset mode from metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        dataset_mode = meta.get('mode', 'pairs')
        logger.info(f"Dataset mode: {dataset_mode}")

        if args.loss == 'cosent' and dataset_mode != 'scored':
            logger.warning("CoSENTLoss requires scored dataset. "
                           "Run 05_prepare_finetune_data.py --mode scored first.")
            logger.warning("Falling back to MNR loss.")
            args.loss = 'mnr'

    # Load datasets
    logger.info("\nLoading datasets...")
    train_dataset = load_from_disk(os.path.join(data_dir, 'train'))
    val_dataset = load_from_disk(os.path.join(data_dir, 'val'))
    val_triplets = load_from_disk(os.path.join(data_dir, 'val_triplets'))

    logger.info(f"Train pairs: {len(train_dataset)}")
    logger.info(f"Val pairs: {len(val_dataset)}")
    logger.info(f"Val triplets: {len(val_triplets)}")
    logger.info(f"Train columns: {train_dataset.column_names}")

    # Load model
    logger.info(f"\nLoading model: {args.model}...")
    model = SentenceTransformer(args.model)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Define loss
    if args.loss == 'cosent':
        loss = CoSENTLoss(model)
        logger.info("Using CoSENTLoss (continuous similarity scores)")
    else:
        loss = MultipleNegativesRankingLoss(model)
        logger.info("Using MultipleNegativesRankingLoss (in-batch negatives)")

    # Define training arguments
    training_args_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=50,
    )

    # NO_DUPLICATES sampler only makes sense for MNR
    if args.loss == 'mnr':
        training_args_kwargs['batch_sampler'] = BatchSamplers.NO_DUPLICATES

    training_args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # Create evaluator
    logger.info("\nSetting up TripletEvaluator...")
    evaluator = TripletEvaluator(
        anchors=val_triplets['anchor'],
        positives=val_triplets['positive'],
        negatives=val_triplets['negative'],
        name="arxiv-val",
    )

    # Evaluate base model first
    logger.info("\nEvaluating base model (before fine-tuning)...")
    base_results = evaluator(model)
    logger.info(f"Base model evaluation: {base_results}")

    # Create trainer
    logger.info("\nStarting training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # Train
    start_time = datetime.now()
    trainer.train()
    duration = (datetime.now() - start_time).total_seconds()

    logger.info(f"\nTraining completed in {duration:.1f}s ({duration/60:.1f} min)")

    # Evaluate fine-tuned model
    logger.info("\nEvaluating fine-tuned model...")
    finetuned_results = evaluator(model)
    logger.info(f"Fine-tuned model evaluation: {finetuned_results}")

    # Save final model
    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    logger.info(f"\nFinal model saved to: {final_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("FINE-TUNING SUMMARY")
    print("=" * 60)
    print(f"Base model:     {args.model}")
    print(f"Loss:           {args.loss}")
    print(f"Training time:  {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Train pairs:    {len(train_dataset)}")
    print(f"Epochs:         {args.epochs}")
    print(f"")
    print(f"Base model eval:       {base_results}")
    print(f"Fine-tuned model eval: {finetuned_results}")
    print(f"")
    print(f"Model saved to: {final_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
