"""
完整的VLMBench Benchmark生成脚本

核心设计理念：
1. Native: 生成原始超图结构 + 查询（基础数据集）
2. Text-Only: 从Native读取，用7种文本模板翻译
3. Vision-Only: 从Native读取，用8种可视化翻译
4. Multimodal: 复用Vision-Only的图片 + 组合文本

数据流：
  Native (生成) → Text-Only (翻译)
                → Vision-Only (翻译)
                → Multimodal (复用Vision-Only图片 + 文本组合)

可视化配置(5种):
  - Native模式: forceatlas2, spring, circular, kamada_kawai, networkx
  - Bipartite模式: bipartite_horizontal, bipartite_radial, hypernetx_style
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random

from config import (
    ALL_TASK_TYPES,
    VISUALIZATION_CONFIGS,
    TEXT_PROMPT_TEMPLATES,
    MULTIMODAL_FIXED_TEXT_TEMPLATE,
    MULTIMODAL_FIXED_VISUALIZATION,
    HYPERGRAPH_GENERATION_CONFIG
)
from tasks import TASK_REGISTRY
from hypergraph_utils import HyperGraph, generate_connected_hypergraph
from hypergraph_constructors import (
    generate_hypergraph_with_cycle,
    generate_hypergraph_with_hamiltonian_path,
    generate_hypergraph_with_coloring
)
from visualization_improved import visualize_hypergraph, visualize_two_hypergraphs
from prompt_generator import generate_text_prompt
from text_encoder import encode_graph
from visualization_instructions_improved import get_instruction


def extract_question_only(question_text):
    """
    从完整的question_text中提取纯问题（去掉图描述）
    
    Args:
        question_text: 完整问题文本（包含图描述）
    
    Returns:
        纯问题文本
    """
    # 查找"Q:"标记，提取其后的内容
    if 'Q:' in question_text:
        parts = question_text.split('Q:', 1)
        if len(parts) > 1:
            question_part = 'Q:' + parts[1]
            # 移除最后的"A: "
            question_part = question_part.replace('\nA: ', '').strip()
            return question_part
    
    # 如果没有Q:标记，返回原文本
    return question_text.strip()


def generate_native_samples(task_type, num_samples, output_dir, size_distribution=None):
    """
    生成Native模式样本：原始超图结构 + 查询（基础数据集）
    这是其他三种模式的数据源
    
    支持断点续传：检查已存在的样本，只生成缺失的部分
    
    Args:
        task_type: 任务类型
        num_samples: 每个任务的样本数量（默认200）
        output_dir: 输出目录 (data/Native/{task_type})
        size_distribution: 规模分布字典 {'small': 0.25, 'medium': 0.5, 'large': 0.25}
    """
    task_class = TASK_REGISTRY[task_type]
    task_dir = Path(output_dir) / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 默认规模分布：small:medium:large = 1:2:1 (25%:50%:25%)
    if size_distribution is None:
        size_distribution = {'small': 0.25, 'medium': 0.5, 'large': 0.25}
    
    # 计算每个规模的样本数
    size_counts = {
        'small': int(num_samples * size_distribution['small']),
        'medium': int(num_samples * size_distribution['medium']),
        'large': int(num_samples * size_distribution['large'])
    }
    # 处理四舍五入误差
    total_allocated = sum(size_counts.values())
    if total_allocated < num_samples:
        size_counts['medium'] += num_samples - total_allocated
    
    # 【断点续传】检查是否已有部分样本
    output_file = task_dir / 'samples.json'
    samples = []
    sample_idx = 0
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        sample_idx = len(samples)
        print(f"  ⚡ 检测到已生成 {sample_idx}/{num_samples} 个样本，继续生成剩余部分...")
        
        if sample_idx >= num_samples:
            print(f"✓ {task_type} Native samples already complete: {output_file}")
            return len(samples)
    
    # 从已生成的样本数开始继续
    remaining = num_samples - sample_idx
    
    # 按规模生成样本（支持断点续传）
    generated_so_far = sample_idx
    for size_name, count in size_counts.items():
        size_config = HYPERGRAPH_GENERATION_CONFIG[size_name]
        
        # 计算该规模已生成和需要生成的数量
        size_start_idx = sum(size_counts[s] for s in ['small', 'medium', 'large'][:['small', 'medium', 'large'].index(size_name)])
        size_end_idx = size_start_idx + count
        
        # 跳过已生成的样本
        if generated_so_far >= size_end_idx:
            continue
        
        skip_count = max(0, generated_so_far - size_start_idx)
        actual_count = count - skip_count
        
        for i in tqdm(range(skip_count, count), desc=f"Generating {task_type} ({size_name})", leave=False, initial=skip_count, total=count):
            # 生成超图
            num_vertices = random.randint(*size_config['num_vertices_range'])
            
            # 根据顶点数和比例计算超边数
            ratio_min, ratio_max = size_config['edge_to_vertex_ratio']
            num_edges_min = max(1, int(num_vertices * ratio_min))
            num_edges_max = int(num_vertices * ratio_max)
            num_edges = random.randint(num_edges_min, num_edges_max)
            
            edge_size_range = size_config['edge_size_range']
            
            # 检查任务是否需要保证有解的构造方法
            guaranteed_info = {}
            if hasattr(task_class, 'use_guaranteed_solution') and task_class.use_guaranteed_solution:
                if task_type == 'StrictHypercycle':
                    # 生成保证有cycle的超图
                    cycle_length = min(num_edges, num_vertices)
                    extra_edges = max(0, num_edges - cycle_length)
                    hypergraph = generate_hypergraph_with_cycle(
                        num_vertices, cycle_length, edge_size_range, extra_edges
                    )
                    # 记录cycle信息（构造时的前cycle_length条边）
                    guaranteed_info['guaranteed_cycle'] = list(range(cycle_length))
                    
                elif task_type == 'HypergraphHamiltonianPath':
                    # 生成保证有Hamilton路径的超图
                    extra_edges = max(0, num_edges - num_vertices)
                    hypergraph, source, target, path_edges = generate_hypergraph_with_hamiltonian_path(
                        num_vertices, edge_size_range, extra_edges
                    )
                    guaranteed_info['guaranteed_path_info'] = {
                        'source': source,
                        'target': target,
                        'path': path_edges
                    }
                    
                elif task_type == 'Hypergraph3Coloring':
                    # 生成保证3-可着色的超图
                    hypergraph, coloring = generate_hypergraph_with_coloring(
                        num_vertices, num_edges, edge_size_range
                    )
                    guaranteed_info['guaranteed_coloring'] = coloring
                    
                else:
                    # 默认生成
                    hypergraph = generate_connected_hypergraph(
                        num_vertices=num_vertices,
                        num_edges=num_edges,
                        edge_size_range=edge_size_range
                    )
            else:
                # 普通生成
                hypergraph = generate_connected_hypergraph(
                    num_vertices=num_vertices,
                    num_edges=num_edges,
                    edge_size_range=edge_size_range
                )
            
            # 准备任务样本（使用N-Set编码生成基础问题）
            sample_data = task_class.prepare_example(hypergraph, encoding_method='N-Set', **guaranteed_info)
            
            # 提取ground_truth和answer
            # 如果ground_truth是字典，使用它；否则使用answer字段
            if 'ground_truth' in sample_data and isinstance(sample_data['ground_truth'], dict):
                ground_truth = sample_data['ground_truth']
            else:
                ground_truth = sample_data['answer']
            
            # 保存超图结构（JSON格式）
            hypergraph_data = {
                'vertices': list(hypergraph.v),
                'hyperedges': [list(edge) for edge in hypergraph.e[0]]
            }
            
            sample = {
                'sample_id': f"{task_type}_native_{sample_idx:04d}",
                'task_type': task_type,
                'hypergraph': hypergraph_data,
                'hypergraph_size': size_name,  # 记录规模
                'question': sample_data['question_text'],
                'ground_truth': ground_truth,
                'metadata': sample_data.get('metadata', {})
            }
            
            # 对于HypergraphIsomorphism任务，保存第二个超图
            if task_type == 'HypergraphIsomorphism' and 'hypergraph_H' in sample_data:
                hg_h = sample_data['hypergraph_H']
                sample['hypergraph_H'] = {
                    'vertices': list(hg_h.v),
                    'hyperedges': [list(edge) for edge in hg_h.e[0]]
                }
                sample['is_isomorphic'] = sample_data.get('is_isomorphic', False)
            
            samples.append(sample)
            sample_idx += 1
            
            # 【断点续传】每生成10个样本自动保存一次
            if sample_idx % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # 最终保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {task_type} Native samples saved: {output_file}")
    return len(samples)


def generate_vision_only_samples(task_type, native_dir, output_dir):
    """
    生成Vision-Only模式样本：从Native读取，用5种可视化翻译
    
    支持断点续传：检查已存在的样本和图片文件，跳过已生成的部分
    
    Args:
        task_type: 任务类型
        native_dir: Native数据目录
        output_dir: 输出目录 (data/Vision-Only/{task_type})
    """
    # 读取Native数据
    native_file = Path(native_dir) / task_type / 'samples.json'
    if not native_file.exists():
        print(f"✗ Native data not found: {native_file}")
        print(f"  Please run Native mode first!")
        return 0
    
    with open(native_file, 'r', encoding='utf-8') as f:
        native_samples = json.load(f)
    
    task_dir = Path(output_dir) / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 【断点续传】加载已生成的样本
    output_file = task_dir / 'samples.json'
    samples = []
    existing_ids = set()
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        existing_ids = {s['sample_id'] for s in samples}
        # print(f"  ⚡ 检测到已生成 {len(samples)} 个样本，跳过已完成部分...")
    
    total_count = len(samples)
    save_counter = 0
    
    for native_sample in tqdm(native_samples, desc=f"Translating {task_type} (Vision-Only)"):
        # 重构超图对象
        hg_data = native_sample['hypergraph']
        hypergraph = HyperGraph(hg_data['vertices'], [tuple(e) for e in hg_data['hyperedges']])
        
        # 对于HypergraphIsomorphism，也重构第二个超图
        hypergraph_H = None
        if task_type == 'HypergraphIsomorphism' and 'hypergraph_H' in native_sample:
            hg_h_data = native_sample['hypergraph_H']
            hypergraph_H = HyperGraph(hg_h_data['vertices'], [tuple(e) for e in hg_h_data['hyperedges']])
        
        # 为每种可视化配置生成样本
        for vis_idx, vis_config in enumerate(VISUALIZATION_CONFIGS):
            sample_id = f"{task_type}_vision_{native_sample['sample_id'].split('_')[-1]}_{vis_idx:02d}"
            
            # 【断点续传】跳过已生成的样本
            if sample_id in existing_ids:
                continue
            
            # 生成可视化图片
            img_filename = f"{sample_id}.png"
            img_path = task_dir / img_filename
            
            # 对于HypergraphIsomorphism使用双超图可视化
            if hypergraph_H is not None:
                visualize_two_hypergraphs(
                    hypergraph_H,
                    hypergraph,
                    output_path=str(img_path),
                    left_title='Hypergraph H',
                    right_title='Hypergraph G',
                    **vis_config
                )
            else:
                visualize_hypergraph(
                    hypergraph,
                    output_path=str(img_path),
                    **vis_config
                )
            
            # 为Vision-Only添加layout-specific可视化说明
            layout = vis_config['layout']
            vision_instruction = get_instruction(layout) + "\n\n"
            
            sample = {
                'sample_id': sample_id,
                'task_type': task_type,
                'native_id': native_sample['sample_id'],  # 关联到Native样本
                'image_path': str(img_path),
                'question': vision_instruction + extract_question_only(native_sample['question']),  # 添加可视化说明
                'ground_truth': native_sample['ground_truth'],
                'visualization_config': vis_config,
                'metadata': native_sample.get('metadata', {}),
                'hypergraph': hg_data  # 添加hypergraph数据用于提取nvertices/nedges
            }
            
            samples.append(sample)
            total_count += 1
            save_counter += 1
            
            # 【断点续传】每生成20个样本自动保存一次
            if save_counter % 20 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # 最终保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {task_type} Vision-Only samples saved: {output_file}")
    print(f"  Base samples: {len(native_samples)} → Total samples: {total_count}")
    return total_count


def generate_text_only_samples(task_type, native_dir, output_dir):
    """
    生成Text-Only模式样本：从Native读取，用7种文本模板翻译
    
    支持断点续传：检查已存在的样本，跳过已生成的部分
    
    Args:
        task_type: 任务类型
        native_dir: Native数据目录
        output_dir: 输出目录 (data/Text-Only/{task_type})
    """
    # 读取Native数据
    native_file = Path(native_dir) / task_type / 'samples.json'
    if not native_file.exists():
        print(f"✗ Native data not found: {native_file}")
        print(f"  Please run Native mode first!")
        return 0
    
    with open(native_file, 'r', encoding='utf-8') as f:
        native_samples = json.load(f)
    
    task_class = TASK_REGISTRY[task_type]
    task_dir = Path(output_dir) / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 【断点续传】加载已生成的样本
    output_file = task_dir / 'samples.json'
    samples = []
    existing_ids = set()
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        existing_ids = {s['sample_id'] for s in samples}
        # print(f"  ⚡ 检测到已生成 {len(samples)} 个样本，跳过已完成部分...")
    
    total_count = len(samples)
    save_counter = 0
    
    for native_sample in tqdm(native_samples, desc=f"Translating {task_type} (Text-Only)"):
        # 重构超图对象
        hg_data = native_sample['hypergraph']
        hypergraph = HyperGraph(hg_data['vertices'], [tuple(e) for e in hg_data['hyperedges']])
        
        # 对于HypergraphIsomorphism，也重构第二个超图
        hypergraph_H = None
        if task_type == 'HypergraphIsomorphism' and 'hypergraph_H' in native_sample:
            hg_h_data = native_sample['hypergraph_H']
            hypergraph_H = HyperGraph(hg_h_data['vertices'], [tuple(e) for e in hg_h_data['hyperedges']])
        
        # 为每种文本prompt模板生成样本
        for prompt_idx, prompt_template in enumerate(TEXT_PROMPT_TEMPLATES):
            sample_id = f"{task_type}_text_{native_sample['sample_id'].split('_')[-1]}_{prompt_idx:02d}"
            
            # 【断点续传】跳过已生成的样本
            if sample_id in existing_ids:
                continue
            
            # 使用该模板生成文本描述
            text_description = encode_graph(hypergraph, prompt_template)
            
            # 对于HypergraphIsomorphism，添加第二个超图的描述
            if hypergraph_H is not None:
                text_description_H = encode_graph(hypergraph_H, prompt_template).replace('G', 'H')
                text_description = (
                    f"There are two hypergraphs: H and G.\n"
                    f"The description of H is: {text_description_H}"
                    f"The description of G is: {text_description}"
                )
            
            sample = {
                'sample_id': sample_id,
                'task_type': task_type,
                'native_id': native_sample['sample_id'],  # 关联到Native样本
                'text_description': text_description,
                'question': extract_question_only(native_sample['question']),  # 使用纯问题
                'ground_truth': native_sample['ground_truth'],
                'prompt_template': prompt_template,
                'metadata': native_sample.get('metadata', {}),
                'hypergraph': hg_data,  # 添加hypergraph结构用于验证
            }
            
            samples.append(sample)
            total_count += 1
            save_counter += 1
            
            # 【断点续传】每生成50个样本自动保存一次
            if save_counter % 50 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # 最终保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {task_type} Text-Only samples saved: {output_file}")
    print(f"  Base samples: {len(native_samples)} → Total samples: {total_count}")
    return total_count


def generate_multimodal_samples(task_type, native_dir, vision_dir, output_dir):
    """
    生成Multimodal模式样本：复用Vision-Only图片 + 组合文本
    
    每个native样本随机生成10个不重复的(文本编码, 可视化)组合
    - 7种文本编码 × 5种可视化 = 35种可能组合
    - 每个native样本随机选择10个组合
    
    支持断点续传：检查已存在的样本，跳过已生成的部分
    
    Args:
        task_type: 任务类型
        native_dir: Native数据目录
        vision_dir: Vision-Only数据目录（复用图片）
        output_dir: 输出目录 (data/Multimodal/{task_type})
    """
    # 读取Native数据
    native_file = Path(native_dir) / task_type / 'samples.json'
    if not native_file.exists():
        print(f"✗ Native data not found: {native_file}")
        print(f"  Please run Native mode first!")
        return 0
    
    with open(native_file, 'r', encoding='utf-8') as f:
        native_samples = json.load(f)
    
    # 读取Vision-Only数据（获取图片路径）
    vision_file = Path(vision_dir) / task_type / 'samples.json'
    if not vision_file.exists():
        print(f"✗ Vision-Only data not found: {vision_file}")
        print(f"  Please run Vision-Only mode first!")
        return 0
    
    with open(vision_file, 'r', encoding='utf-8') as f:
        vision_samples = json.load(f)
    
    # 建立Vision-Only样本的索引：(native_id, vis_config) -> vision_sample
    vision_index = {}
    for vs in vision_samples:
        native_id = vs['native_id']
        vis_config = vs['visualization_config']
        vis_key = (vis_config['layout'], vis_config.get('mode', 'default'))
        vision_index[(native_id, vis_key)] = vs
    
    task_class = TASK_REGISTRY[task_type]
    task_dir = Path(output_dir) / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 【断点续传】加载已生成的样本
    output_file = task_dir / 'samples.json'
    samples = []
    existing_ids = set()
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        existing_ids = {s['sample_id'] for s in samples}
        print(f"  ⚡ 检测到已生成 {len(samples)} 个样本，跳过已完成部分...")
    
    total_count = len(samples)
    save_counter = 0
    
    for native_sample in tqdm(native_samples, desc=f"Combining {task_type} (Multimodal)"):
        native_id = native_sample['sample_id']
        base_idx = native_id.split('_')[-1]
        
        # 重构超图对象（用于生成文本）
        hg_data = native_sample['hypergraph']
        hypergraph = HyperGraph(hg_data['vertices'], [tuple(e) for e in hg_data['hyperedges']])
        
        # 生成所有 7种文本编码 × 5种可视化 = 35种组合
        combo_idx = 0
        for prompt_template in TEXT_PROMPT_TEMPLATES:
            for vis_config in VISUALIZATION_CONFIGS:
                sample_id = f"{task_type}_mm_{base_idx}_{combo_idx:02d}"
                
                # 【断点续传】跳过已生成的样本
                if sample_id in existing_ids:
                    combo_idx += 1
                    continue
                
                # 查找对应的Vision-Only图片
                vis_key = (vis_config['layout'], vis_config.get('mode', 'default'))
                vision_sample = vision_index.get((native_id, vis_key))
                
                if vision_sample is None:
                    print(f"Warning: Vision sample not found for {native_id}, {vis_key}")
                    combo_idx += 1
                    continue
                
                # 生成该模板的文本描述
                text_description = encode_graph(hypergraph, prompt_template)
                
                # 添加可视化说明
                layout = vis_config['layout']
                vision_instruction = get_instruction(layout) + "\n\n"
                
                sample = {
                    'sample_id': sample_id,
                    'task_type': task_type,
                    'mode': 'multimodal',
                    'native_id': native_id,
                    'image_path': vision_sample['image_path'],  # 复用图片
                    'text_description': text_description,
                    'question': vision_instruction + extract_question_only(native_sample['question']),
                    'ground_truth': native_sample['ground_truth'],
                    'visualization_config': vis_config,
                    'prompt_template': prompt_template,
                    'metadata': native_sample.get('metadata', {}),
                    'hypergraph': hg_data
                }
                
                samples.append(sample)
                total_count += 1
                save_counter += 1
                combo_idx += 1
                
                # 【断点续传】每生成50个样本自动保存一次
                if save_counter % 50 == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # 最终保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {task_type} Multimodal samples saved: {output_file}")
    print(f"  Native: {len(native_samples)} → Multimodal: {total_count}")
    print(f"  组合: 7种文本编码 × 5种可视化 = 35种组合/native样本")
    print(f"  预期总数: {len(native_samples) * 35}, 实际: {total_count}")
    print(f"  Images reused from Vision-Only (no new images generated)")
    return total_count


def main():
    parser = argparse.ArgumentParser(description='VLMBench完整Benchmark生成器')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['native', 'vision-only', 'text-only', 'multimodal', 'all'],
                        help='生成模式')
    
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='指定任务类型（默认：所有任务）')
    
    parser.add_argument('--num_samples', type=int, default=200,
                        help='每个任务的Native样本数量（默认：200，将按规模分布生成）')
    
    parser.add_argument('--output_base', type=str, default='./data',
                        help='输出基础目录（默认：./data）')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认：42）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 确定要生成的任务
    tasks = args.tasks if args.tasks else ALL_TASK_TYPES
    
    # 验证任务类型
    for task in tasks:
        if task not in TASK_REGISTRY:
            print(f"错误: 未知任务类型 '{task}'")
            print(f"可用任务: {', '.join(ALL_TASK_TYPES)}")
            return
    
    print("=" * 70)
    print(f"VLMBench完整Benchmark生成")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"任务数量: {len(tasks)}")
    print(f"任务列表: {', '.join(tasks)}")
    print(f"Native样本数/任务: {args.num_samples} (Small:Medium:Large = 1:2:1)")
    print(f"输出目录: {args.output_base}")
    print(f"随机种子: {args.seed}")
    print()
    print(f"预期输出:")
    print(f"  Native: {args.num_samples} × {len(tasks)} = {args.num_samples * len(tasks)}")
    print(f"  Text-Only: {args.num_samples} × 7编码 × {len(tasks)} = {args.num_samples * 7 * len(tasks)}")
    print(f"  Vision-Only: {args.num_samples} × 5可视化 × {len(tasks)} = {args.num_samples * 5 * len(tasks)}")
    print(f"  Multimodal: {args.num_samples} × 35组合 × {len(tasks)} = {args.num_samples * 35 * len(tasks)}")
    print("=" * 70)
    
    total_samples = 0
    
    # 定义目录
    native_dir = Path(args.output_base) / 'Native'
    vision_dir = Path(args.output_base) / 'Vision-Only'
    text_dir = Path(args.output_base) / 'Text-Only'
    multimodal_dir = Path(args.output_base) / 'Multimodal'
    
    # 根据模式生成样本
    if args.mode == 'native' or args.mode == 'all':
        print("\n[1/4] 生成Native模式样本（原始数据）...")
        print(f"规模分布: Small(25%) + Medium(50%) + Large(25%)")
        for task in tasks:
            count = generate_native_samples(task, args.num_samples, native_dir)
            total_samples += count
    
    if args.mode == 'vision-only' or args.mode == 'all':
        print("\n[2/4] 翻译为Vision-Only模式（从Native读取）...")
        for task in tasks:
            count = generate_vision_only_samples(task, native_dir, vision_dir)
            total_samples += count
    
    if args.mode == 'text-only' or args.mode == 'all':
        print("\n[3/4] 翻译为Text-Only模式（从Native读取）...")
        for task in tasks:
            count = generate_text_only_samples(task, native_dir, text_dir)
            total_samples += count
    
    if args.mode == 'multimodal' or args.mode == 'all':
        print("\n[4/4] 组合Multimodal模式（复用Vision-Only图片）...")
        for task in tasks:
            count = generate_multimodal_samples(task, native_dir, vision_dir, multimodal_dir)
            total_samples += count
    
    print("\n" + "=" * 70)
    print("生成完成！")
    print("=" * 70)
    print(f"总样本数: {total_samples}")
    print(f"输出位置: {args.output_base}")
    
    # 统计信息
    if args.mode == 'all':
        expected_per_task = args.num_samples * (1 + 7 + 5 + 35)  # native + text + vision + multimodal
        print(f"\n预期样本数/任务: {expected_per_task}")
        print(f"  - Native: {args.num_samples}")
        print(f"  - Text-Only: {args.num_samples * 7}")
        print(f"  - Vision-Only: {args.num_samples * 5}")
        print(f"  - Multimodal: {args.num_samples * 35}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
