"""
Text Prompt Generator

为超图生成不同编码风格的文本描述
"""

import numpy as np
from hypergraph_utils import HyperGraph


def generate_n_set_encoding(hypergraph):
    """
    N-Set编码：Node-Set表示法
    
    例如:
    Hypergraph with vertices: {v0, v1, v2, v3, v4}
    Hyperedges:
    - e0: {v0, v1, v2}
    - e1: {v1, v2, v3}
    - e2: {v2, v3, v4}
    """
    vertices = sorted(hypergraph.vertices)
    
    # 格式化顶点（添加v前缀）
    vertex_names = [f"v{v}" for v in vertices]
    text = f"Hypergraph with {len(vertices)} vertices: {{{', '.join(vertex_names)}}}\n"
    text += f"Hyperedges ({len(hypergraph.hyperedges)}):\n"
    
    for i, edge in enumerate(hypergraph.hyperedges):
        edge_names = [f"v{v}" for v in sorted(edge)]
        edge_str = "{" + ", ".join(edge_names) + "}"
        text += f"- e{i}: {edge_str}\n"
    
    return text.strip()


def generate_ho_inc_encoding(hypergraph):
    """
    HO-Inc编码：High-Order Incidence表示法
    
    例如:
    Vertices: v0, v1, v2, v3, v4
    
    Vertex v0 is in hyperedges: e0
    Vertex v1 is in hyperedges: e0, e1
    Vertex v2 is in hyperedges: e0, e1, e2
    ...
    """
    vertices = sorted(hypergraph.vertices)
    vertex_names = [f"v{v}" for v in vertices]
    
    text = f"Vertices: {', '.join(vertex_names)}\n\n"
    
    for v in vertices:
        v_name = f"v{v}"
        incident_edges = []
        for i, edge in enumerate(hypergraph.hyperedges):
            if v in edge:
                incident_edges.append(f"e{i}")
        
        if incident_edges:
            text += f"Vertex {v_name} is in hyperedges: {', '.join(incident_edges)}\n"
        else:
            text += f"Vertex {v_name} is isolated (not in any hyperedge)\n"
    
    return text.strip()


def generate_ho_neigh_encoding(hypergraph):
    """
    HO-Neigh编码：High-Order Neighborhood表示法
    
    例如:
    v0's neighbors: {v1, v2} (via e0)
    v1's neighbors: {v0, v2} (via e0), {v2, v3} (via e1)
    ...
    """
    vertices = sorted(hypergraph.vertices)
    
    text = ""
    
    for v in vertices:
        v_name = f"v{v}"
        neighbors_info = []
        
        for i, edge in enumerate(hypergraph.hyperedges):
            if v in edge:
                neighbors = sorted(edge - {v})
                if neighbors:
                    neighbor_names = [f"v{n}" for n in neighbors]
                    neighbors_str = "{" + ", ".join(neighbor_names) + "}"
                    neighbors_info.append(f"{neighbors_str} (via e{i})")
        
        if neighbors_info:
            text += f"{v_name}'s neighbors: {', '.join(neighbors_info)}\n"
        else:
            text += f"{v_name} has no neighbors\n"
    
    return text.strip()


def generate_n_pair_encoding(hypergraph):
    """
    N-Pair编码：Node-Pair表示法 (clique expansion)
    
    将每个超边展开为所有节点对
    例如: e0: {v0, v1, v2} → (v0, v1), (v0, v2), (v1, v2)
    """
    vertices = sorted(hypergraph.vertices)
    vertex_names = [f"v{v}" for v in vertices]
    
    text = f"Vertices: {', '.join(vertex_names)}\n\n"
    text += "Edges (as node pairs):\n"
    
    for i, edge in enumerate(hypergraph.hyperedges):
        sorted_edge = sorted(edge)
        pairs = []
        
        for j in range(len(sorted_edge)):
            for k in range(j + 1, len(sorted_edge)):
                pairs.append(f"(v{sorted_edge[j]}, v{sorted_edge[k]})")
        
        if pairs:
            text += f"From e{i}: {', '.join(pairs)}\n"
    
    return text.strip()


def generate_lo_inc_encoding(hypergraph):
    """
    LO-Inc编码：Low-Order Incidence表示法
    
    只列出超边及其包含的节点，简化版本
    """
    text = f"Hypergraph with {len(hypergraph.vertices)} vertices and {len(hypergraph.hyperedges)} hyperedges.\n\n"
    
    for i, edge in enumerate(hypergraph.hyperedges):
        edge_names = [f"v{v}" for v in sorted(edge)]
        edge_str = ", ".join(edge_names)
        text += f"e{i} contains: {edge_str}\n"
    
    return text.strip()


def generate_inc_mat_encoding(hypergraph):
    """
    Inc-Mat编码：Incidence Matrix表示法
    
    构建关联矩阵：行=顶点，列=超边
    """
    vertices = sorted(hypergraph.vertices)
    
    text = "Incidence Matrix (rows=vertices, columns=hyperedges):\n\n"
    
    # 表头
    text += "       "
    for i in range(len(hypergraph.hyperedges)):
        text += f"e{i}  "
    text += "\n"
    
    # 矩阵内容
    for v in vertices:
        v_name = f"v{v}"
        text += f"{v_name:4s}   "
        for edge in hypergraph.hyperedges:
            text += " 1  " if v in edge else " 0  "
        text += "\n"
    
    return text.strip()


def generate_adj_mat_encoding(hypergraph):
    """
    Adj-Mat编码：Adjacency Matrix表示法
    
    基于超边展开的邻接矩阵（clique expansion）
    如果两个顶点在同一个超边中，则视为相邻
    """
    vertices = sorted(hypergraph.vertices)
    n = len(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # 构建邻接矩阵
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for edge in hypergraph.hyperedges:
        edge_list = list(edge)
        for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):
                v1, v2 = edge_list[i], edge_list[j]
                idx1, idx2 = v_to_idx[v1], v_to_idx[v2]
                adj_matrix[idx1][idx2] = 1
                adj_matrix[idx2][idx1] = 1
    
    # 格式化输出
    text = "Adjacency Matrix (based on clique expansion):\n\n"
    
    # 表头
    text += "       "
    for v in vertices:
        v_name = f"v{v}"
        text += f"{v_name:4s} "
    text += "\n"
    
    # 矩阵内容
    for i, v in enumerate(vertices):
        v_name = f"v{v}"
        text += f"{v_name:4s}   "
        for j in range(n):
            text += f" {adj_matrix[i][j]}  "
        text += "\n"
    
    return text.strip()


def generate_text_prompt(hypergraph, question, template='N-Set'):
    """
    根据模板生成文本prompt
    
    Args:
        hypergraph: HyperGraph对象
        question: 问题文本
        template: 编码模板名称
    
    Returns:
        完整的文本prompt
    """
    # 生成超图描述
    if template == 'N-Set':
        graph_desc = generate_n_set_encoding(hypergraph)
    elif template == 'HO-Inc':
        graph_desc = generate_ho_inc_encoding(hypergraph)
    elif template == 'HO-Neigh':
        graph_desc = generate_ho_neigh_encoding(hypergraph)
    elif template == 'N-Pair':
        graph_desc = generate_n_pair_encoding(hypergraph)
    elif template == 'LO-Inc':
        graph_desc = generate_lo_inc_encoding(hypergraph)
    elif template == 'Inc-Mat':
        graph_desc = generate_inc_mat_encoding(hypergraph)
    elif template == 'Adj-Mat':
        graph_desc = generate_adj_mat_encoding(hypergraph)
    else:
        raise ValueError(f"Unknown template: {template}")
    
    # 组合成完整prompt
    prompt = f"{graph_desc}\n\nQuestion: {question}"
    
    return prompt


def test_all_templates():
    """测试所有模板"""
    from config import TEXT_PROMPT_TEMPLATES
    
    # 创建测试超图
    vertices = [0, 1, 2, 3, 4]
    hyperedges = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4)
    ]
    hg = HyperGraph(vertices, hyperedges)
    
    question = "How many vertices are in this hypergraph?"
    
    print("=" * 70)
    print("Testing All Text Prompt Templates")
    print("=" * 70)
    
    for template in TEXT_PROMPT_TEMPLATES:
        print(f"\n[{template}]")
        print("-" * 70)
        prompt = generate_text_prompt(hg, question, template)
        print(prompt)
        print()


if __name__ == '__main__':
    test_all_templates()
