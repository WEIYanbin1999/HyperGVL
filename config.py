"""
VLMBench Configuration Module

定义所有的任务类型、可视化选项和文本描述风格的配置
"""

# ========== 任务类型定义 ==========

# 基础理解型任务 (Basic Understanding Tasks) - 聚焦超图核心结构与阶特性
BASIC_UNDERSTANDING_TASKS = [
    'VertexCount',                    # 顶点数量统计
    'HyperedgeCount',                 # 超边数量统计
    'SpecificVertexDegreeCount',      # 特定degree的顶点数统计
    'SpecificOrderHyperedgeCount',    # 特定order的超边数统计
    'DirectNeighbor',                 # 直接邻居识别
    'NNeighbor',                      # order≥n约束下的邻居识别
]

# 高阶推理任务 (Advanced Reasoning Tasks) - 基于order的算法级推理
ADVANCED_REASONING_TASKS = [
    'OrderBasedShortestPath',         # 超边阶为权的最短路径
    'OrderBasedMaxFlow',              # 超边阶为权的最大流
    'StrictHypercycle',              # 最大超环检测
    'HypergraphIsomorphism',          # 超图同构识别
    'Hypergraph3Coloring',          # 超图3着色
    'HypergraphHamiltonianPath',      # 超图哈密顿路径
]

# 所有任务列表
ALL_TASKS = BASIC_UNDERSTANDING_TASKS + ADVANCED_REASONING_TASKS
ALL_TASK_TYPES = ALL_TASKS  # 别名，用于兼容

# 任务类别映射
TASK_CATEGORIES = {
    'basic_understanding': BASIC_UNDERSTANDING_TASKS,
    'advanced_reasoning': ADVANCED_REASONING_TASKS,
}

# 任务难度级别
TASK_DIFFICULTY = {
    # Basic Understanding - Level 1-2
    'VertexCount': 1,
    'HyperedgeCount': 1,
    'SpecificVertexDegreeCount': 2,
    'SpecificOrderHyperedgeCount': 2,
    'DirectNeighbor': 2,
    'NNeighbor': 2,
    
    # Advanced Reasoning - Level 3-4
    'OrderBasedShortestPath': 3,
    'OrderBasedMaxFlow': 4,
    'StrictHypercycle': 3,
    'HypergraphIsomorphism': 4,
    'Hypergraph3Coloring': 4,
    'HypergraphHamiltonianPath': 4,
}

# ========== 可视化配置 ==========
# 基于实验验证，删除了VLM无法理解的彩色凸包可视化
# 保留和新增VLM可理解的显式表示方法

# 经过验证有效的可视化方法：
# 1. bipartite_horizontal, bipartite_radial (实验证明100%准确率)
# 2. labeled_regions (带标签的彩色区域 - 新增)
# 3. edge_list (边列表显式展示 - 新增)
# 4. star_expansion (星形扩展 - 新增)
# 5. annotated_network (边标注网络 - 新增)

EFFECTIVE_VISUALIZATION_LAYOUTS = [
    'bipartite_horizontal',  # 验证有效：100%准确率
    'shell',                 # 验证有效：100%准确率
    'star_expansion',        # 验证有效：100%准确率
    'annotated_network',     # 验证有效：100%准确率
    'networkx',              # 原生风格（测试中）
]

# 所有可视化风格（使用改进后的方法）
ALL_VISUALIZATION_LAYOUTS = EFFECTIVE_VISUALIZATION_LAYOUTS

# 可视化模式分类
VISUALIZATION_MODES = {
    'explicit_nodes': ['bipartite_horizontal', 'bipartite_radial', 'star_expansion'],
    'labeled': ['labeled_regions', 'annotated_network'],
    'list': ['edge_list'],
}

# ========== 文本描述风格 ==========

TEXT_ENCODING_STYLES = [
    'N-Set',        # Node-Set表示法
    'HO-Inc',       # High-Order Incidence表示法
    'HO-Neigh',     # High-Order Neighborhood表示法
    'N-Pair',       # Node-Pair表示法 (clique expansion)
    'LO-Inc',       # Low-Order Incidence表示法
    'Inc-Mat',      # Incidence Matrix表示法
    'Adj-Mat',      # Adjacency Matrix表示法
]

# 文本Prompt模板（用于Text-Only和Multimodal模式）
TEXT_PROMPT_TEMPLATES = TEXT_ENCODING_STYLES

# Multimodal模式的固定文本模板（默认使用N-Set）
MULTIMODAL_FIXED_TEXT_TEMPLATE = 'N-Set'

# Multimodal模式的固定可视化配置（使用验证有效的bipartite_horizontal）
MULTIMODAL_FIXED_VISUALIZATION = {
    'layout': 'bipartite_horizontal',
}

# ========== 可视化配置列表 ==========

# 为Vision-Only和Multimodal模式定义的8种可视化配置
# 可视化配置列表（用于生成benchmark数据）
# 每个配置生成一套可视化样本
VISUALIZATION_CONFIGS = [
    {'layout': 'bipartite_horizontal'},   # 验证：100%准确率
    {'layout': 'shell'},                  # 验证：100%准确率
    {'layout': 'star_expansion'},         # 验证：100%准确率
    {'layout': 'annotated_network'},      # 验证：100%准确率
    {'layout': 'networkx'},               # 原生风格（测试中）
]

# ========== VQA 模式配置 ==========

VQA_MODES = {
    'text_only': {
        'use_text': True,
        'use_image': False,
        'description': '仅基于文本描述的VQA'
    },
    'vision_only': {
        'use_text': False,
        'use_image': True,
        'description': '仅基于可视化图像的VQA'
    },
    'multimodal': {
        'use_text': True,
        'use_image': True,
        'description': '同时包含文本描述和可视化图像的VQA'
    }
}

# ========== 超图生成配置 ==========

HYPERGRAPH_GENERATION_CONFIG = {
    'small': {
        'num_vertices_range': (5, 10),
        'edge_to_vertex_ratio': (0.2, 1.5),  # 超边数 = 顶点数 × ratio
        'edge_size_range': (2, 4),
    },
    'medium': {
        'num_vertices_range': (10, 15),
        'edge_to_vertex_ratio': (0.2, 1.5),
        'edge_size_range': (2, 5),
    },
    'large': {
        'num_vertices_range': (15, 20),
        'edge_to_vertex_ratio': (0.2, 1.5),
        'edge_size_range': (2, 6),
    }
}

# ========== 默认配置 ==========

DEFAULT_CONFIG = {
    'task_types': ALL_TASKS,
    'visualization_mode': 'native',
    'visualization_layout': 'forceatlas2',
    'text_encoding': 'N-Set',
    'vqa_mode': 'multimodal',
    'hypergraph_size': 'medium',
    'random_seed': 42,
    'num_samples_per_task': 100,
}

# ========== 实验配置模板 ==========

EXPERIMENT_TEMPLATES = {
    'quick_test': {
        'task_types': BASIC_UNDERSTANDING_TASKS[:2],
        'num_samples_per_task': 10,
        'vqa_mode': 'multimodal',
    },
    'basic_benchmark': {
        'task_types': BASIC_UNDERSTANDING_TASKS,
        'num_samples_per_task': 50,
        'vqa_mode': 'multimodal',
    },
    'comprehensive_benchmark': {
        'task_types': ALL_TASKS,
        'num_samples_per_task': 100,
        'vqa_mode': 'multimodal',
    },
    'text_only_benchmark': {
        'task_types': ALL_TASKS,
        'num_samples_per_task': 100,
        'vqa_mode': 'text_only',
    },
    'vision_only_benchmark': {
        'task_types': ALL_TASKS,
        'num_samples_per_task': 100,
        'vqa_mode': 'vision_only',
    }
}


def get_config(template_name=None, **kwargs):
    """
    获取配置
    
    Args:
        template_name: 实验模板名称
        **kwargs: 覆盖的配置项
    
    Returns:
        配置字典
    """
    if template_name and template_name in EXPERIMENT_TEMPLATES:
        config = {**DEFAULT_CONFIG, **EXPERIMENT_TEMPLATES[template_name]}
    else:
        config = DEFAULT_CONFIG.copy()
    
    # 使用kwargs覆盖配置
    config.update(kwargs)
    
    return config


def validate_config(config):
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
    
    Returns:
        (is_valid, error_message)
    """
    # 验证任务类型
    for task in config.get('task_types', []):
        if task not in ALL_TASKS:
            return False, f"Unknown task type: {task}"
    
    # 验证可视化布局
    layout = config.get('visualization_layout')
    if layout and layout not in ALL_VISUALIZATION_LAYOUTS:
        return False, f"Unknown visualization layout: {layout}"
    
    # 验证文本编码
    encoding = config.get('text_encoding')
    if encoding and encoding not in TEXT_ENCODING_STYLES:
        return False, f"Unknown text encoding: {encoding}"
    
    # 验证VQA模式
    vqa_mode = config.get('vqa_mode')
    if vqa_mode and vqa_mode not in VQA_MODES:
        return False, f"Unknown VQA mode: {vqa_mode}"
    
    return True, "Configuration is valid"


def print_config(config):
    """打印配置信息"""
    print("=" * 60)
    print("VLMBench Configuration")
    print("=" * 60)
    
    print(f"\nTask Types ({len(config.get('task_types', []))}):")
    for task in config.get('task_types', []):
        category = [k for k, v in TASK_CATEGORIES.items() if task in v][0]
        difficulty = TASK_DIFFICULTY.get(task, 'N/A')
        print(f"  - {task} (Category: {category}, Difficulty: {difficulty})")
    
    print(f"\nVisualization:")
    print(f"  - Mode: {config.get('visualization_mode', 'N/A')}")
    print(f"  - Layout: {config.get('visualization_layout', 'N/A')}")
    
    print(f"\nText Encoding: {config.get('text_encoding', 'N/A')}")
    print(f"VQA Mode: {config.get('vqa_mode', 'N/A')}")
    print(f"Hypergraph Size: {config.get('hypergraph_size', 'N/A')}")
    print(f"Samples per Task: {config.get('num_samples_per_task', 'N/A')}")
    print(f"Random Seed: {config.get('random_seed', 'N/A')}")
    
    print("=" * 60)
