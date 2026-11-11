"""
VLMBench Text Encoder Module

提供多种文本描述风格来编码超图
"""

import random
from typing import Dict, List, Tuple
from hypergraph_utils import HyperGraph


# ========== 节点和边的命名字典 ==========

def create_integer_name_dict(max_count: int = 1000) -> Dict[int, str]:
    """创建整数命名字典"""
    return {i: str(i) for i in range(max_count)}


NODE_ENCODER_DICT = {
    "N-Pair": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
    "LO-Inc": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
    "Adj-Mat": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
    "N-Set": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
    "HO-Inc": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
    "Inc-Mat": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
    "HO-Neigh": {k: 'v' + str(v) for k, v in create_integer_name_dict().items()},
}

EDGES_ENCODER_DICT = {
    "N-Pair": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
    "LO-Inc": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
    "Adj-Mat": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
    "N-Set": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
    "HO-Inc": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
    "Inc-Mat": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
    "HO-Neigh": {k: 'e' + str(v) for k, v in create_integer_name_dict().items()},
}


# ========== 辅助函数 ==========

def create_vertex_string(name_dict: Dict[int, str], nvertices: int) -> str:
    """创建顶点列表的字符串描述"""
    vertex_string = ""
    for i in range(nvertices - 1):
        vertex_string += name_dict[i] + ", "
    vertex_string += "and " + name_dict[nvertices - 1]
    return vertex_string


def create_hyperedge_string(edge_dict: Dict[int, str], nedges: int) -> str:
    """创建超边列表的字符串描述"""
    edge_string = ""
    for i in range(nedges - 1):
        edge_string += edge_dict[i] + ", "
    edge_string += "and " + edge_dict[nedges - 1]
    return edge_string


# ========== 编码器实现 ==========

def N_Set_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    N-Set编码器：将超图编码为集合表示法
    
    示例: G describes a hypergraph among vertices v0, v1, and v2 and among hyperedges e0 and e1.
          The hyperedges in G are: (v0,v1),(v1,v2).
    """
    output = (
        "In an undirected hypergraph, (i, j, k) means that vertex i, vertex j and vertex k are "
        "connected with an undirected hyperedge. "
    )
    
    vertices_string = create_vertex_string(name_dict, len(graph.v))
    edges_string = create_hyperedge_string(edge_dict, len(graph.e[0]))
    
    output += f"G describes a hypergraph among vertices {vertices_string} "
    output += f"and among hyperedges {edges_string}.\n"
    
    if graph.e[0]:
        output += "The hyperedges in G are: "
        for edge in graph.e[0]:
            tmp = ''
            for i in edge:
                tmp += f"{name_dict[i]},"
            tmp = '(' + tmp[:-1] + '),'
            output += tmp
    
    return output.strip() + ".\n"


def HO_Inc_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    HO-Inc编码器：高阶关联列表表示法
    
    示例: G describes a hypergraph among vertices v0, v1, v2 and among hyperedges e0, e1.
          In this hypergraph:
          vertex v0 is connected to vertex v1 with hyperedge e0.
          vertex v1 is connected to vertices v0, v2 with hyperedge e0, to vertex v2 with hyperedge e1.
    """
    num_v = len(graph.v)
    vertices_string = create_vertex_string(name_dict, num_v)
    edges_string = create_hyperedge_string(edge_dict, len(graph.e[0]))
    
    output = f"G describes a hypergraph among vertices {vertices_string} and among hyperedges {edges_string}.\n"
    
    if graph.e[0]:
        output += "In this hypergraph:\n"
    
    for source_vertex in range(num_v):
        neighbor_edges = graph.edges(source_vertex)
        output += f"vertex {name_dict[source_vertex]} is connected"
        
        for e in neighbor_edges:
            edge = graph.e[0][e]
            target_vertices = [v for v in edge if v != source_vertex]
            
            if len(target_vertices) > 1:
                target_str = ", ".join([name_dict[v] for v in target_vertices])
                output += f" to vertices {target_str} with hyperedge {edge_dict[e]},"
            elif len(target_vertices) == 1:
                output += f" to vertex {name_dict[target_vertices[0]]} with hyperedge {edge_dict[e]},"
        
        output = output[:-1] + ".\n"
    
    return output


def HO_Neigh_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    HO-Neigh编码器：高阶邻域表示法
    
    示例: G describes a hypergraph among vertices v0, v1 and hyperedges e0, e1.
          In this hypergraph:
          vertex v0 is connected to hyperedges e0,e1.
          vertex v1 is connected to hyperedge e0.
          Hyperedge e0 is connected to vertices v0,v1.
    """
    vertices_string = create_vertex_string(name_dict, len(graph.v))
    edge_string = create_hyperedge_string(edge_dict, len(graph.e[0]))
    
    output = f"G describes a hypergraph among vertices {vertices_string} and hyperedges {edge_string}.\n"
    
    if graph.e[0]:
        output += "In this hypergraph:\n"
    
    # 顶点到超边的连接
    for source_vertex in graph.v:
        incident_edges = []
        for j, edge in enumerate(graph.e[0]):
            if source_vertex in edge:
                incident_edges.append(j)
        
        if len(incident_edges) > 1:
            output += f"vertex {name_dict[source_vertex]} is connected to hyperedges "
            for i in incident_edges:
                output += f'{edge_dict[i]},'
            output = output[:-1] + '.\n'
        elif len(incident_edges) == 1:
            output += f"vertex {name_dict[source_vertex]} is connected to hyperedge {edge_dict[incident_edges[0]]}.\n"
    
    # 超边到顶点的连接
    for k, source_edge in enumerate(graph.e[0]):
        output += f'Hyperedge {edge_dict[k]} is connected to vertices '
        for n in source_edge:
            output += f'{name_dict[n]},'
        output = output[:-1] + '.\n'
    
    return output


def N_Pair_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    N-Pair编码器：节点对表示法（基于clique expansion）
    
    示例: In an undirected hypergraph, (i,j) means that vertex i and vertex j are
          connected with an undirected hyperedge.
          G describes a hypergraph among vertices v0, v1, v2.
          The edges in G are: (v0, v1), (v1, v2).
    """
    num_v, edges = graph.clique_expansion()
    vertices_string = create_vertex_string(name_dict, num_v)
    
    output = (
        "In an undirected hypergraph, (i,j) means that vertex i and vertex j are "
        "connected with an undirected hyperedge. "
    )
    
    output += f"G describes a hypergraph among vertices {vertices_string}.\n"
    
    if edges:
        output += "The edges in G are: "
        for edge in edges:
            output += f"({name_dict[edge[0]]}, {name_dict[edge[1]]}), "
        output = output[:-2] + ".\n"
    
    return output


def LO_Inc_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    LO-Inc编码器：低阶关联列表（基于clique expansion）
    
    示例: G describes a hypergraph among vertices v0, v1, v2.
          In this hypergraph:
          vertex v0 is connected to vertex v1.
          vertex v1 is connected to vertices v0, v2.
    """
    num_v, edges = graph.clique_expansion()
    vertices_string = create_vertex_string(name_dict, num_v)
    
    output = f"G describes a hypergraph among vertices {vertices_string}.\n"
    
    if edges:
        output += "In this hypergraph:\n"
        
        # 构建邻接表
        adjacency = {i: [] for i in range(num_v)}
        for v1, v2 in edges:
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        for source_vertex in range(num_v):
            neighbors = sorted(set(adjacency[source_vertex]))
            if neighbors:
                output += f"vertex {name_dict[source_vertex]} is connected to "
                if len(neighbors) > 1:
                    neighbor_str = ", ".join([name_dict[n] for n in neighbors])
                    output += f"vertices {neighbor_str}.\n"
                else:
                    output += f"vertex {name_dict[neighbors[0]]}.\n"
    
    return output


def Inc_Mat_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    Inc-Mat编码器：关联矩阵表示法
    
    示例: G describes a hypergraph with incidence matrix:
              e0  e1
          v0  1   0
          v1  1   1
          v2  0   1
    """
    num_v = len(graph.v)
    num_e = len(graph.e[0])
    
    output = "G describes a hypergraph with incidence matrix:\n"
    
    # 表头
    output += "    "
    for j in range(num_e):
        output += f"{edge_dict[j]:^4}"
    output += "\n"
    
    # 矩阵内容
    for i in range(num_v):
        output += f"{name_dict[i]:^4}"
        for j in range(num_e):
            if i in graph.e[0][j]:
                output += "  1 "
            else:
                output += "  0 "
        output += "\n"
    
    return output


def Adj_Mat_encoder(graph: HyperGraph, name_dict: Dict[int, str], edge_dict: Dict[int, str]) -> str:
    """
    Adj-Mat编码器：邻接矩阵表示法（基于clique expansion）
    
    示例: G describes a hypergraph with adjacency matrix:
              v0  v1  v2
          v0  0   1   0
          v1  1   0   1
          v2  0   1   0
    """
    num_v = len(graph.v)
    
    # 构建邻接矩阵
    adj_matrix = [[0] * num_v for _ in range(num_v)]
    
    for edge in graph.e[0]:
        for i in edge:
            for j in edge:
                if i != j:
                    adj_matrix[i][j] = 1
    
    output = "G describes a hypergraph with adjacency matrix:\n"
    
    # 表头
    output += "    "
    for j in range(num_v):
        output += f"{name_dict[j]:^4}"
    output += "\n"
    
    # 矩阵内容
    for i in range(num_v):
        output += f"{name_dict[i]:^4}"
        for j in range(num_v):
            output += f"{adj_matrix[i][j]:^4}"
        output += "\n"
    
    return output


# ========== 编码器映射 ==========

ENCODER_FUNCTIONS = {
    'N-Set': N_Set_encoder,
    'HO-Inc': HO_Inc_encoder,
    'HO-Neigh': HO_Neigh_encoder,
    'N-Pair': N_Pair_encoder,
    'LO-Inc': LO_Inc_encoder,
    'Inc-Mat': Inc_Mat_encoder,
    'Adj-Mat': Adj_Mat_encoder,
}


# ========== 主编码函数 ==========

def encode_graph(graph: HyperGraph, encoding_method: str = 'N-Set') -> str:
    """
    将超图编码为文本描述
    
    Args:
        graph: HyperGraph实例
        encoding_method: 编码方法，可选值见ENCODER_FUNCTIONS的键
        
    Returns:
        文本描述字符串
    """
    if encoding_method not in ENCODER_FUNCTIONS:
        raise ValueError(f"Unknown encoding method: {encoding_method}. "
                        f"Available methods: {list(ENCODER_FUNCTIONS.keys())}")
    
    node_dict = NODE_ENCODER_DICT[encoding_method]
    edge_dict = EDGES_ENCODER_DICT[encoding_method]
    
    encoder_func = ENCODER_FUNCTIONS[encoding_method]
    return encoder_func(graph, node_dict, edge_dict)


def get_available_encoders() -> List[str]:
    """获取所有可用的编码器名称"""
    return list(ENCODER_FUNCTIONS.keys())
