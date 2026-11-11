"""
超图构造器 - 生成保证有特定性质的超图
用于StrictHypercycle和HypergraphHamiltonianPath等任务
"""
import random
from typing import List, Tuple
from hypergraph_utils import HyperGraph


def generate_hypergraph_with_cycle(num_vertices: int, cycle_length: int, 
                                   edge_size_range: Tuple[int, int],
                                   extra_edges: int = 0) -> HyperGraph:
    """
    生成保证包含strict hypercycle的超图
    
    策略：先构造一个环，然后添加额外边
    
    Args:
        num_vertices: 顶点数
        cycle_length: 环的长度（超边数）
        edge_size_range: 超边大小范围 (min, max)
        extra_edges: 额外添加的超边数
        
    Returns:
        包含strict hypercycle的超图
    """
    if cycle_length < 3:
        cycle_length = 3
    if cycle_length > num_vertices:
        cycle_length = num_vertices
    
    vertices = list(range(num_vertices))
    edges = []
    
    # 1. 构造一个严格的hypercycle
    # 选择cycle_length个顶点形成环
    cycle_vertices = random.sample(vertices, cycle_length)
    
    # 为环上的每条边创建超边，确保相邻边恰好共享1个顶点
    for i in range(cycle_length):
        curr_v = cycle_vertices[i]
        next_v = cycle_vertices[(i + 1) % cycle_length]
        
        # 创建包含当前顶点和下一个顶点的超边
        edge_size = random.randint(max(2, edge_size_range[0]), edge_size_range[1])
        
        # 必须包含curr_v（与前一条边共享）和next_v（与后一条边共享）
        # 但要确保不包含其他cycle顶点，否则会破坏"恰好1个"的约束
        edge = {curr_v, next_v}
        
        # 添加非cycle顶点填充到edge_size
        non_cycle_vertices = [v for v in vertices if v not in cycle_vertices]
        if len(edge) < edge_size and non_cycle_vertices:
            additional = random.sample(non_cycle_vertices, min(edge_size - len(edge), len(non_cycle_vertices)))
            edge.update(additional)
        
        edges.append(tuple(sorted(edge)))
    
    # 2. 添加额外的随机边（不破坏原有cycle）
    for _ in range(extra_edges):
        edge_size = random.randint(edge_size_range[0], edge_size_range[1])
        edge = tuple(sorted(random.sample(vertices, min(edge_size, num_vertices))))
        if edge not in edges:
            edges.append(edge)
    
    return HyperGraph(vertices, edges)


def generate_hypergraph_with_hamiltonian_path(num_vertices: int, 
                                               edge_size_range: Tuple[int, int],
                                               extra_edges: int = 0) -> Tuple[HyperGraph, int, int, List[int]]:
    """
    生成保证包含Hamilton路径的超图
    
    策略：构造一条路径，保证能访问所有顶点恰好一次
    
    Args:
        num_vertices: 顶点数
        edge_size_range: 超边大小范围 (min, max)
        extra_edges: 额外添加的超边数
        
    Returns:
        (超图, 起点, 终点, 路径边索引列表)
    """
    vertices = list(range(num_vertices))
    edges = []
    
    # 1. 构造Hamilton路径
    # 随机打乱顶点顺序作为访问顺序
    visit_order = random.sample(vertices, num_vertices)
    source = visit_order[0]
    target = visit_order[-1]
    
    # 构造路径：创建num_vertices条超边，每条边覆盖路径上的连续顶点
    path_edges = []
    for i in range(num_vertices):
        curr_v = visit_order[i]
        
        # 决定这条边的大小
        edge_size = random.randint(max(2, edge_size_range[0]), edge_size_range[1])
        
        # 边必须包含当前顶点和下一个顶点（如果有）
        if i < num_vertices - 1:
            next_v = visit_order[i + 1]
            edge = [curr_v, next_v]
        else:
            edge = [curr_v]
        
        # 添加已访问过的顶点填充边（不破坏Hamilton性质）
        visited = visit_order[:i+1]
        available = [v for v in visited if v not in edge]
        
        if len(edge) < edge_size and available:
            additional = random.sample(available, min(edge_size - len(edge), len(available)))
            edge.extend(additional)
        
        # 如果还不够，从未访问的顶点中选择（但不能是下一个顶点）
        if len(edge) < edge_size and i < num_vertices - 1:
            unvisited = [v for v in visit_order[i+2:] if v not in edge]
            if unvisited:
                additional = random.sample(unvisited, min(edge_size - len(edge), len(unvisited)))
                edge.extend(additional)
        
        edges.append(tuple(sorted(edge)))
        path_edges.append(len(edges) - 1)
    
    # 2. 添加额外的随机边
    for _ in range(extra_edges):
        edge_size = random.randint(edge_size_range[0], edge_size_range[1])
        edge = tuple(sorted(random.sample(vertices, min(edge_size, num_vertices))))
        if edge not in edges:
            edges.append(edge)
    
    hypergraph = HyperGraph(vertices, edges)
    return hypergraph, source, target, path_edges


def generate_hypergraph_with_coloring(num_vertices: int, num_edges: int,
                                      edge_size_range: Tuple[int, int]) -> Tuple[HyperGraph, dict]:
    """
    生成保证3-可着色的超图
    
    策略：先为顶点分配颜色，再生成满足约束的超边
    
    Args:
        num_vertices: 顶点数
        num_edges: 超边数
        edge_size_range: 超边大小范围
        
    Returns:
        (超图, 着色方案字典)
    """
    vertices = list(range(num_vertices))
    
    # 1. 随机为顶点分配3种颜色
    coloring = {v: random.randint(0, 2) for v in vertices}
    
    edges = []
    
    # 2. 生成满足"每条边至少2种颜色"的超边
    for _ in range(num_edges):
        edge_size = random.randint(edge_size_range[0], edge_size_range[1])
        edge_size = min(edge_size, num_vertices)
        
        # 策略：至少选择2种不同颜色的顶点
        max_attempts = 100
        for attempt in range(max_attempts):
            edge_vertices = random.sample(vertices, edge_size)
            edge_colors = set(coloring[v] for v in edge_vertices)
            
            # 确保至少有2种颜色
            if len(edge_colors) >= 2:
                edge = tuple(sorted(edge_vertices))
                if edge not in edges:
                    edges.append(edge)
                    break
        else:
            # 如果随机失败，强制构造
            # 选择2种不同颜色
            color1, color2 = random.sample([0, 1, 2], 2)
            vertices_c1 = [v for v in vertices if coloring[v] == color1]
            vertices_c2 = [v for v in vertices if coloring[v] == color2]
            
            if vertices_c1 and vertices_c2:
                # 至少各选1个
                edge_vertices = [random.choice(vertices_c1), random.choice(vertices_c2)]
                
                # 填充到edge_size
                remaining = [v for v in vertices if v not in edge_vertices]
                if len(edge_vertices) < edge_size and remaining:
                    additional = random.sample(remaining, min(edge_size - len(edge_vertices), len(remaining)))
                    edge_vertices.extend(additional)
                
                edge = tuple(sorted(edge_vertices))
                if edge not in edges:
                    edges.append(edge)
    
    hypergraph = HyperGraph(vertices, edges)
    return hypergraph, coloring
