"""
Embedding Utilities

Helper functions for loading models, generating embeddings, and managing
embedding storage. Follows patterns from OpenAlex/scripts/generate_embeddings_fast.py
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Manages embedding model loading and batch generation.
    Supports GPU acceleration and memory-efficient processing.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 128,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name
            batch_size: Number of texts to process at once
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing embedding generator for {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {batch_size}")

        # Lazy loading - model loaded on first use
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self._model.to(self.device)
            logger.info(f"Model loaded successfully (device: {self.device})")
        return self._model

    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts")

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                device=self.device,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )

        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def generate_embeddings_batched(
        self,
        texts: List[str],
        checkpoint_callback: Optional[callable] = None,
        checkpoint_interval: int = 10000
    ) -> np.ndarray:
        """
        Generate embeddings with checkpointing for long-running operations.

        Args:
            texts: List of text strings
            checkpoint_callback: Optional function called after each checkpoint
                                with (embeddings_so_far, current_index)
            checkpoint_interval: Save checkpoint every N texts

        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])

        num_texts = len(texts)
        embedding_dim = self.get_embedding_dim()
        all_embeddings = np.zeros((num_texts, embedding_dim), dtype=np.float32)

        logger.info(f"Generating embeddings for {num_texts} texts with checkpointing")

        for start_idx in tqdm(range(0, num_texts, checkpoint_interval), desc="Checkpoints"):
            end_idx = min(start_idx + checkpoint_interval, num_texts)
            batch_texts = texts[start_idx:end_idx]

            # Generate embeddings for this checkpoint
            batch_embeddings = self.generate_embeddings(
                batch_texts,
                show_progress=False
            )

            all_embeddings[start_idx:end_idx] = batch_embeddings

            # Call checkpoint callback
            if checkpoint_callback:
                checkpoint_callback(all_embeddings[:end_idx], end_idx)

            # Clear GPU cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        logger.info(f"Completed embedding generation: {all_embeddings.shape}")
        return all_embeddings

    def clear_model(self):
        """Clear model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            logger.info("Model cleared from memory")


def load_model(
    model_name: str,
    batch_size: int = 128,
    device: Optional[str] = None
) -> EmbeddingGenerator:
    """
    Load an embedding model.

    Args:
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        device: Device to use ('cuda', 'cpu', or None for auto)

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )


def save_embeddings(
    embeddings: np.ndarray,
    model_name: str,
    output_dir: str,
    metadata: Optional[Dict] = None
):
    """
    Save embeddings to disk.

    Args:
        embeddings: NumPy array of embeddings
        model_name: Model name (for directory naming)
        output_dir: Base output directory
        metadata: Optional metadata dictionary to save
    """
    # Create model-specific directory
    safe_model_name = model_name.replace('/', '-').replace('\\', '-')
    model_dir = os.path.join(output_dir, safe_model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save embeddings as .npy
    embeddings_path = os.path.join(model_dir, 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        'model_name': model_name,
        'shape': embeddings.shape,
        'dtype': str(embeddings.dtype),
        'saved_at': str(np.datetime64('now'))
    })

    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")


def load_embeddings(
    model_name: str,
    data_dir: str
) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings from disk.

    Args:
        model_name: Model name
        data_dir: Base data directory

    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    safe_model_name = model_name.replace('/', '-').replace('\\', '-')
    model_dir = os.path.join(data_dir, safe_model_name)

    # Load embeddings
    embeddings_path = os.path.join(model_dir, 'embeddings.npy')
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings from {embeddings_path} with shape {embeddings.shape}")

    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")

    return embeddings, metadata


def embeddings_exist(model_name: str, data_dir: str) -> bool:
    """
    Check if embeddings already exist for a model.

    Args:
        model_name: Model name
        data_dir: Base data directory

    Returns:
        True if embeddings exist, False otherwise
    """
    safe_model_name = model_name.replace('/', '-').replace('\\', '-')
    embeddings_path = os.path.join(data_dir, safe_model_name, 'embeddings.npy')
    return os.path.exists(embeddings_path)


def get_available_models(data_dir: str) -> List[str]:
    """
    Get list of models that have embeddings available.

    Args:
        data_dir: Base data directory

    Returns:
        List of model names
    """
    if not os.path.exists(data_dir):
        return []

    models = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            embeddings_path = os.path.join(item_path, 'embeddings.npy')
            if os.path.exists(embeddings_path):
                # Convert safe name back to model name
                model_name = item.replace('-', '/')
                models.append(model_name)

    return models


def combine_text_for_embedding(title: str, abstract: Optional[str]) -> str:
    """
    Combine title and abstract for embedding generation.

    Args:
        title: Paper title
        abstract: Paper abstract (optional)

    Returns:
        Combined text string
    """
    if abstract:
        return f"{title}. {abstract}"
    return title


def calculate_embedding_stats(embeddings: np.ndarray) -> Dict:
    """
    Calculate statistics about embeddings.

    Args:
        embeddings: NumPy array of embeddings

    Returns:
        Dictionary with statistics
    """
    return {
        'shape': embeddings.shape,
        'mean': float(np.mean(embeddings)),
        'std': float(np.std(embeddings)),
        'min': float(np.min(embeddings)),
        'max': float(np.max(embeddings)),
        'norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'norm_std': float(np.std(np.linalg.norm(embeddings, axis=1))),
    }


if __name__ == '__main__':
    # Test embedding utilities
    print("Testing Embedding Utilities")
    print("=" * 60)

    # Test texts
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text."
    ]

    # Test with a small model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nLoading model: {model_name}")

    generator = load_model(model_name, batch_size=2)
    print(f"Embedding dimension: {generator.get_embedding_dim()}")

    # Generate embeddings
    print(f"\nGenerating embeddings for {len(test_texts)} texts...")
    embeddings = generator.generate_embeddings(test_texts, show_progress=True)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")

    # Calculate stats
    stats = calculate_embedding_stats(embeddings)
    print(f"\nEmbedding statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test save/load
    print(f"\nTesting save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_embeddings(embeddings, model_name, tmpdir, metadata={'test': True})
        loaded_embeddings, loaded_metadata = load_embeddings(model_name, tmpdir)
        print(f"Loaded shape: {loaded_embeddings.shape}")
        print(f"Metadata: {loaded_metadata}")
        assert np.allclose(embeddings, loaded_embeddings), "Embeddings don't match!"
        print("✓ Save/load test passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
