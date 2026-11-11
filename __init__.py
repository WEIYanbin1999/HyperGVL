"""
VLMBench - Comprehensive Hypergraph Understanding Benchmark for Vision-Language Models

A comprehensive benchmark system designed to evaluate Vision-Language Models (VLMs) 
on hypergraph understanding and reasoning capabilities.
"""

__version__ = '1.0.0'
__author__ = 'VLMBench Team'

from .config import (
    get_config,
    validate_config,
    print_config,
    BASIC_UNDERSTANDING_TASKS,
    LOCAL_INTEGRATION_TASKS,
    GLOBAL_REASONING_TASKS,
    COMPLEX_ALGORITHM_TASKS,
    ALL_TASKS,
    VQA_MODES,
)

from .hypergraph_utils import (
    HyperGraph,
    generate_random_hypergraph,
    generate_connected_hypergraph,
)

from .tasks import (
    get_task,
    get_all_tasks,
)

from .text_encoder import (
    encode_graph,
    get_available_encoders,
)

from .visualization import (
    visualize_hypergraph,
    visualize_two_hypergraphs,
    get_available_layouts,
    HypergraphVisualizer,
)

from .benchmark_generator import (
    BenchmarkGenerator,
)

from .evaluator import (
    VLMEvaluator,
)

__all__ = [
    # Config
    'get_config',
    'validate_config',
    'print_config',
    'BASIC_UNDERSTANDING_TASKS',
    'LOCAL_INTEGRATION_TASKS',
    'GLOBAL_REASONING_TASKS',
    'COMPLEX_ALGORITHM_TASKS',
    'ALL_TASKS',
    'VQA_MODES',
    
    # Hypergraph Utils
    'HyperGraph',
    'generate_random_hypergraph',
    'generate_connected_hypergraph',
    
    # Tasks
    'get_task',
    'get_all_tasks',
    
    # Text Encoder
    'encode_graph',
    'get_available_encoders',
    
    # Visualization
    'visualize_hypergraph',
    'visualize_two_hypergraphs',
    'get_available_layouts',
    'HypergraphVisualizer',
    
    # Benchmark Generator
    'BenchmarkGenerator',
    
    # Evaluator
    'VLMEvaluator',
]
