"""
生成二分类任务的Native数据，带标签平衡
针对HypergraphIsomorphism, Hypergraph3Coloring, HypergraphHamiltonianPath
"""
import json
import random
from pathlib import Path
from tqdm import tqdm
from hypergraph_utils import generate_connected_hypergraph, HyperGraph
from text_encoder import encode_graph
from config import HYPERGRAPH_GENERATION_CONFIG

random.seed(42)

def generate_isomorphism_sample(hypergraph, target_label, encoding_method='N-Set'):
    """生成HypergraphIsomorphism样本，控制标签"""
    if target_label:  # 生成同构的
        graph_text_G = encode_graph(hypergraph, encoding_method)
        
        # 随机打乱顶点编号
        vertices = list(hypergraph.v)
        shuffled_vertices = vertices.copy()
        random.shuffle(shuffled_vertices)
        vertex_map = {old: new for old, new in zip(vertices, shuffled_vertices)}
        
        shuffled_edges = []
        for edge in hypergraph.e[0]:
            shuffled_edge = tuple(sorted([vertex_map[v] for v in edge]))
            shuffled_edges.append(shuffled_edge)
        
        hypergraph_H = HyperGraph(shuffled_vertices, shuffled_edges)
        graph_text_H = encode_graph(hypergraph_H, encoding_method)
        answer = "Yes"
    else:  # 生成非同构的
        graph_text_G = encode_graph(hypergraph, encoding_method)
        
        num_vertices = len(hypergraph.v)
        edge_degrees = [len(edge) for edge in hypergraph.e[0]]
        
        new_edges = set()
        max_attempts = 100
        attempts = 0
        for degree in edge_degrees:
            while attempts < max_attempts:
                edge = tuple(sorted(random.sample(range(num_vertices), degree)))
                if edge not in new_edges:
                    new_edges.add(edge)
                    break
                attempts += 1
        
        hypergraph_H = HyperGraph(list(range(num_vertices)), list(new_edges))
        graph_text_H = encode_graph(hypergraph_H, encoding_method)
        answer = "No"
    
    graph_text_H = graph_text_H.replace('G', 'H')
    question_text = (
        f"There are two hypergraphs: H and G.\n"
        f"The description of H is: {graph_text_H}"
        f"The description of G is: {graph_text_G}"
        f"Q: Are these two hypergraphs isomorphic? "
        f"(Two hypergraphs are isomorphic if there exists a vertex relabeling that transforms one into the other). "
        f"List the answer after 'Ans:' in the format [Yes/No].\nA: "
    )
    
    return {
        'question': question_text,
        'ground_truth': answer,
        'hypergraph_H': {
            'vertices': list(hypergraph_H.v),
            'hyperedges': [list(edge) for edge in hypergraph_H.e[0]]
        }
    }

def generate_balanced_isomorphism():
    """生成HypergraphIsomorphism任务"""
    print("\n" + "="*70)
    print("生成 HypergraphIsomorphism (带标签平衡)")
    print("="*70)
    
    task_type = 'HypergraphIsomorphism'
    output_dir = Path('data/Native')
    task_dir = output_dir / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    
    size_counts = {'small': 50, 'medium': 100, 'large': 50}
    all_samples = []
    sample_idx = 0
    
    for size_name, count in size_counts.items():
        print(f"\n生成 {size_name} 规模...")
        size_config = HYPERGRAPH_GENERATION_CONFIG[size_name]
        
        # 平衡生成
        target_per_label = count // 2
        
        for label in [True, False]:
            label_name = "Positive" if label else "Negative"
            for i in tqdm(range(target_per_label), desc=f"{size_name} {label_name}"):
                # 生成超图
                num_vertices = random.randint(*size_config['num_vertices_range'])
                ratio_min, ratio_max = size_config['edge_to_vertex_ratio']
                num_edges = random.randint(
                    max(1, int(num_vertices * ratio_min)),
                    int(num_vertices * ratio_max)
                )
                edge_size_range = size_config['edge_size_range']
                
                hypergraph = generate_connected_hypergraph(num_vertices, num_edges, edge_size_range)
                sample_data = generate_isomorphism_sample(hypergraph, label)
                
                hypergraph_data = {
                    'vertices': list(hypergraph.v),
                    'hyperedges': [list(edge) for edge in hypergraph.e[0]]
                }
                
                sample = {
                    'sample_id': f"{task_type}_native_{sample_idx:04d}",
                    'task_type': task_type,
                    'hypergraph': hypergraph_data,
                    'hypergraph_size': size_name,
                    'question': sample_data['question'],
                    'ground_truth': sample_data['ground_truth'],
                    'metadata': {'hypergraph_H': sample_data['hypergraph_H']}
                }
                
                all_samples.append(sample)
                sample_idx += 1
    
    # 保存
    output_file = task_dir / 'samples.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ {task_type} 已保存: {output_file}")
    print(f"  总样本数: {len(all_samples)}")
    
    # 验证标签分布
    label_dist = {}
    for size in ['small', 'medium', 'large']:
        size_samples = [s for s in all_samples if s['hypergraph_size'] == size]
        yes_count = sum(1 for s in size_samples if s['ground_truth'] == 'Yes')
        no_count = len(size_samples) - yes_count
        label_dist[size] = (yes_count, no_count)
    
    print(f"\n标签分布验证:")
    for size, (yes, no) in label_dist.items():
        print(f"  {size}: Yes={yes}, No={no}")

if __name__ == '__main__':
    generate_balanced_isomorphism()
    print("\n" + "="*70)
    print("完成!")
    print("="*70)
