"""
Configuration file for ArXiv paper analysis system.
Contains all constants, paths, and model definitions.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
DB_PATH = os.path.join(DATA_DIR, 'arxiv_papers.db')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PCA_VIZ_DIR = os.path.join(RESULTS_DIR, 'pca_visualizations')
CLUSTER_REPORTS_DIR = os.path.join(RESULTS_DIR, 'cluster_reports')

# ============================================================================
# ARXIV API SETTINGS
# ============================================================================
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
RATE_LIMIT_DELAY = 3.0  # Seconds between requests (ArXiv requirement)
MAX_RESULTS_PER_REQUEST = 1000  # ArXiv API maximum
REQUEST_TIMEOUT = 30  # Seconds

# Retry settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5

# ============================================================================
# TARGET CATEGORIES AND COUNTS
# ============================================================================
# ALL ArXiv categories from https://arxiv.org/category_taxonomy
# 118 categories total (6 aliases excluded: cs.NA→math.NA, cs.SY→eess.SY,
# math.IT→cs.IT, math.MP→math-ph, q-fin.EC→econ.GN, stat.TH→math.ST)

ARXIV_CATEGORIES = {
    # Astrophysics (6 categories)
    'astro-ph.CO': 200,  # Cosmology and Nongalactic Astrophysics
    'astro-ph.EP': 200,  # Earth and Planetary Astrophysics
    'astro-ph.GA': 200,  # Astrophysics of Galaxies
    'astro-ph.HE': 200,  # High Energy Astrophysical Phenomena
    'astro-ph.IM': 200,  # Instrumentation and Methods for Astrophysics
    'astro-ph.SR': 200,  # Solar and Stellar Astrophysics

    # Computer Science (40 categories)
    'cs.AI': 200,   # Artificial Intelligence
    'cs.AR': 200,   # Hardware Architecture
    'cs.CC': 200,   # Computational Complexity
    'cs.CE': 200,   # Computational Engineering, Finance, and Science
    'cs.CG': 200,   # Computational Geometry
    'cs.CL': 200,   # Computation and Language
    'cs.CR': 200,   # Cryptography and Security
    'cs.CV': 200,   # Computer Vision and Pattern Recognition
    'cs.CY': 200,   # Computers and Society
    'cs.DB': 200,   # Databases
    'cs.DC': 200,   # Distributed, Parallel, and Cluster Computing
    'cs.DL': 200,   # Digital Libraries
    'cs.DM': 200,   # Discrete Mathematics
    'cs.DS': 200,   # Data Structures and Algorithms
    'cs.ET': 200,   # Emerging Technologies
    'cs.FL': 200,   # Formal Languages and Automata Theory
    'cs.GL': 200,   # General Literature
    'cs.GR': 200,   # Graphics
    'cs.GT': 200,   # Computer Science and Game Theory
    'cs.HC': 200,   # Human-Computer Interaction
    'cs.IR': 200,   # Information Retrieval
    'cs.IT': 200,   # Information Theory
    'cs.LG': 200,   # Machine Learning
    'cs.LO': 200,   # Logic in Computer Science
    'cs.MA': 200,   # Multiagent Systems
    'cs.MM': 200,   # Multimedia
    'cs.MS': 200,   # Mathematical Software
    'cs.NE': 200,   # Neural and Evolutionary Computing
    'cs.NI': 200,   # Networking and Internet Architecture
    'cs.OH': 200,   # Other Computer Science
    'cs.OS': 200,   # Operating Systems
    'cs.PF': 200,   # Performance
    'cs.PL': 200,   # Programming Languages
    'cs.RO': 200,   # Robotics
    'cs.SC': 200,   # Symbolic Computation
    'cs.SD': 200,   # Sound
    'cs.SE': 200,   # Software Engineering
    'cs.SI': 200,   # Social and Information Networks

    # Economics (3 categories)
    'econ.EM': 200,  # Econometrics
    'econ.GN': 200,  # General Economics
    'econ.TH': 200,  # Theoretical Economics

    # Electrical Engineering and Systems Science (4 categories)
    'eess.AS': 200,  # Audio and Speech Processing
    'eess.IV': 200,  # Image and Video Processing
    'eess.SP': 200,  # Signal Processing
    'eess.SY': 200,  # Systems and Control

    # Physics - General (5 categories)
    'gr-qc': 200,     # General Relativity and Quantum Cosmology
    'hep-ex': 200,    # High Energy Physics - Experiment
    'hep-lat': 200,   # High Energy Physics - Lattice
    'hep-ph': 200,    # High Energy Physics - Phenomenology
    'hep-th': 200,    # High Energy Physics - Theory
    'math-ph': 200,   # Mathematical Physics
    'nucl-ex': 200,   # Nuclear Experiment
    'nucl-th': 200,   # Nuclear Theory
    'quant-ph': 200,  # Quantum Physics

    # Mathematics (28 categories)
    'math.AC': 200,  # Commutative Algebra
    'math.AG': 200,  # Algebraic Geometry
    'math.AP': 200,  # Analysis of PDEs
    'math.AT': 200,  # Algebraic Topology
    'math.CA': 200,  # Classical Analysis and ODEs
    'math.CO': 200,  # Combinatorics
    'math.CT': 200,  # Category Theory
    'math.CV': 200,  # Complex Variables
    'math.DG': 200,  # Differential Geometry
    'math.DS': 200,  # Dynamical Systems
    'math.FA': 200,  # Functional Analysis
    'math.GM': 200,  # General Mathematics
    'math.GN': 200,  # General Topology
    'math.GR': 200,  # Group Theory
    'math.GT': 200,  # Geometric Topology
    'math.HO': 200,  # History and Overview
    'math.KT': 200,  # K-Theory and Homology
    'math.LO': 200,  # Logic
    'math.MG': 200,  # Metric Geometry
    'math.NA': 200,  # Numerical Analysis
    'math.NT': 200,  # Number Theory
    'math.OA': 200,  # Operator Algebras
    'math.OC': 200,  # Optimization and Control
    'math.PR': 200,  # Probability
    'math.QA': 200,  # Quantum Algebra
    'math.RA': 200,  # Rings and Algebras
    'math.RT': 200,  # Representation Theory
    'math.SG': 200,  # Symplectic Geometry
    'math.SP': 200,  # Spectral Theory
    'math.ST': 200,  # Statistics Theory

    # Nonlinear Sciences (5 categories)
    'nlin.AO': 200,  # Adaptation and Self-Organizing Systems
    'nlin.CD': 200,  # Chaotic Dynamics
    'nlin.CG': 200,  # Cellular Automata and Lattice Gases
    'nlin.PS': 200,  # Pattern Formation and Solitons
    'nlin.SI': 200,  # Exactly Solvable and Integrable Systems

    # Quantitative Biology (10 categories)
    'q-bio.BM': 200,  # Biomolecules
    'q-bio.CB': 200,  # Cell Behavior
    'q-bio.GN': 200,  # Genomics
    'q-bio.MN': 200,  # Molecular Networks
    'q-bio.NC': 200,  # Neurons and Cognition
    'q-bio.OT': 200,  # Other Quantitative Biology
    'q-bio.PE': 200,  # Populations and Evolution
    'q-bio.QM': 200,  # Quantitative Methods
    'q-bio.SC': 200,  # Subcellular Processes
    'q-bio.TO': 200,  # Tissues and Organs

    # Quantitative Finance (8 categories)
    'q-fin.CP': 200,  # Computational Finance
    'q-fin.GN': 200,  # General Finance
    'q-fin.MF': 200,  # Mathematical Finance
    'q-fin.PM': 200,  # Portfolio Management
    'q-fin.PR': 200,  # Pricing of Securities
    'q-fin.RM': 200,  # Risk Management
    'q-fin.ST': 200,  # Statistical Finance
    'q-fin.TR': 200,  # Trading and Market Microstructure

    # Statistics (5 categories)
    'stat.AP': 200,  # Applications
    'stat.CO': 200,  # Computation
    'stat.ME': 200,  # Methodology
    'stat.ML': 200,  # Machine Learning
    'stat.OT': 200,  # Other Statistics
}

# Total: 118 categories × 25 papers = 2,950 papers
# Comprehensive coverage across ALL ArXiv domains!

# Date range for fetching (optional - can be None for all time)
FETCH_START_DATE = '2020-01-01'  # Fetch papers from 2020 onwards for recency
FETCH_END_DATE = None  # None = up to present

# ============================================================================
# EMBEDDING MODELS (MTEB Benchmark)
# ============================================================================
# Format: 'model_name': dimension
EMBEDDING_MODELS = {
    # Top performers (1024-dim)
    'BAAI/bge-large-en-v1.5': 1024,
    'Alibaba-NLP/gte-large-en-v1.5': 1024,
    'Qwen/Qwen3-Embedding-0.6B': 1024,
    'Qwen/Qwen3-Embedding-4B': 2560,

    # Large models (4096-dim) - generated on Google Colab
    'intfloat/e5-mistral-7b-instruct': 4096,

    # Baseline models (768-dim)
    'sentence-transformers/all-mpnet-base-v2': 768,
    'BAAI/bge-base-en-v1.5': 768,
    'nomic-ai/nomic-embed-text-v1.5': 768,
    'allenai-specter': 768,  # Scientific paper specialist
}

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================
# Batch sizes by embedding dimension (GPU memory optimization)
BATCH_SIZE_BY_DIM = {
    768: 256,    # Small models
    1024: 128,   # Medium models
    4096: 32,    # Large models
}

# Default batch size
BATCH_SIZE_DEFAULT = 128

# Checkpoint settings for long-running operations
CHECKPOINT_INTERVAL = 10000  # Save checkpoint every N papers

# GPU settings
GPU_MEMORY_LIMIT = 0.9  # Use 90% of available GPU memory
USE_GPU_IF_AVAILABLE = True

# Database batch size for bulk inserts
DB_BATCH_SIZE = 1000

# ============================================================================
# CLUSTERING SETTINGS
# ============================================================================
# PCA dimensions
PCA_COMPONENTS_2D = 2
PCA_COMPONENTS_3D = 3

# K-means cluster range for elbow method
MIN_CLUSTERS = 5
MAX_CLUSTERS = 20

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Plot dimensions
VIZ_DPI = 300
VIZ_WIDTH = 1200
VIZ_HEIGHT = 800

# Color schemes
COLOR_PALETTE = 'Set2'  # Seaborn/Matplotlib palette
PLOTLY_TEMPLATE = 'plotly_dark'  # Plotly theme

# Point size in scatter plots
SCATTER_POINT_SIZE = 5

# Export formats
VIZ_EXPORT_FORMATS = ['html', 'png']  # Plotly exports

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
# Metrics to calculate
CALCULATE_CATEGORY_PURITY = True
CALCULATE_SILHOUETTE_SCORE = True
CALCULATE_DAVIES_BOULDIN = True
CALCULATE_PCA_VARIANCE = True

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# DATABASE SCHEMA SQL
# ============================================================================
DB_SCHEMA = """
-- Main papers table
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    categories TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    published_date TEXT,
    updated_date TEXT,
    doi TEXT,
    arxiv_url TEXT,
    pdf_url TEXT,
    comment TEXT,
    journal_ref TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_primary_category ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_published_date ON papers(published_date);

-- Embedding models metadata
CREATE TABLE IF NOT EXISTS embedding_models (
    model_name TEXT PRIMARY KEY,
    dimension INTEGER NOT NULL,
    generated_at TIMESTAMP,
    num_papers_embedded INTEGER,
    avg_time_per_paper REAL,
    notes TEXT
);

-- Evaluation results
CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    category_purity_overall REAL,
    silhouette_score REAL,
    davies_bouldin_score REAL,
    num_clusters INTEGER,
    pca_variance_explained REAL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detailed_results TEXT,
    FOREIGN KEY (model_name) REFERENCES embedding_models(model_name)
);
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_batch_size(model_name: str) -> int:
    """Get optimal batch size for a model based on its dimension."""
    dimension = EMBEDDING_MODELS.get(model_name)
    if dimension is None:
        return BATCH_SIZE_DEFAULT
    return BATCH_SIZE_BY_DIM.get(dimension, BATCH_SIZE_DEFAULT)


def get_embedding_path(model_name: str) -> str:
    """Get the directory path for a model's embeddings."""
    # Sanitize model name for filesystem
    safe_name = model_name.replace('/', '-').replace('\\', '-')
    return os.path.join(EMBEDDINGS_DIR, safe_name)


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        EMBEDDINGS_DIR,
        RESULTS_DIR,
        PCA_VIZ_DIR,
        CLUSTER_REPORTS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


if __name__ == '__main__':
    # Print configuration summary
    print("ArXiv Paper Analysis Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Database: {DB_PATH}")
    print(f"Target Papers: {sum(ARXIV_CATEGORIES.values())}")
    print(f"Categories: {list(ARXIV_CATEGORIES.keys())}")
    print(f"Embedding Models: {len(EMBEDDING_MODELS)}")
    print(f"GPU Enabled: {USE_GPU_IF_AVAILABLE}")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()
    print("All directories created successfully!")
