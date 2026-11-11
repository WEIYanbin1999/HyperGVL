"""
VLMBench Hypergraph Utilities

提供超图生成、操作和转换的基础工具
"""

import random
import copy
import itertools
import numpy as np
from typing import List, Tuple, Set, Dict, Optional


class HyperGraph:
    """超图数据结构"""
    
    def __init__(self, vertices: List[int], hyperedges: List[Tuple[int, ...]]):
        """
        初始化超图
        
        Args:
            vertices: 顶点列表
            hyperedges: 超边列表，每个超边是一个顶点元组
        """
        self.v = list(vertices)
        self.e = [list(hyperedges)]  # 保持与原有格式兼容
        self.num_vertices = len(vertices)
        self.num_edges = len(hyperedges)
        
    @property
    def vertices(self):
        """返回顶点集合（用于外部访问）"""
        return set(self.v)
    
    @property
    def hyperedges(self):
        """返回超边列表（用于外部访问）"""
        return [set(edge) for edge in self.e[0]]
        
    def __repr__(self):
        return f"HyperGraph({self.num_vertices} vertices, {self.num_edges} hyperedges)"
    
    def edges(self, vertex: int) -> List[int]:
        """
        获取包含指定顶点的所有超边索引
        
        Args:
            vertex: 顶点ID
            
        Returns:
            超边索引列表
        """
        edge_indices = []
        for idx, edge in enumerate(self.e[0]):
            if vertex in edge:
                edge_indices.append(idx)
        return edge_indices
    
    def neighbors(self, vertex: int) -> Set[int]:
        """
        获取指定顶点的邻居顶点（在同一超边中的其他顶点）
        
        Args:
            vertex: 顶点ID
            
        Returns:
            邻居顶点集合
        """
        neighbors = set()
        for edge in self.e[0]:
            if vertex in edge:
                neighbors.update(edge)
        neighbors.discard(vertex)  # 移除自己
        return neighbors
    
    def vertex_degree(self, vertex: int) -> int:
        """
        获取顶点的度（包含该顶点的超边数量）
        
        Args:
            vertex: 顶点ID
            
        Returns:
            顶点度数
        """
        return len(self.edges(vertex))
    
    def degree(self, vertex: int) -> int:
        """vertex_degree的别名"""
        return self.vertex_degree(vertex)
    
    def hyperedge_degree(self, edge_idx: int) -> int:
        """
        获取超边的度（超边中顶点的数量）
        
        Args:
            edge_idx: 超边索引
            
        Returns:
            超边度数
        """
        if 0 <= edge_idx < len(self.e[0]):
            return len(self.e[0][edge_idx])
        return 0
    
    def clique_expansion(self) -> Tuple[int, List[Tuple[int, int]]]:
        """
        将超图转换为clique expansion图（将每个超边扩展为完全子图）
        
        Returns:
            (顶点数, 边列表)
        """
        edges = []
        for hyperedge in self.e[0]:
            # 为每个超边生成所有可能的顶点对
            for i, v1 in enumerate(hyperedge):
                for v2 in hyperedge[i+1:]:
                    edge = tuple(sorted([v1, v2]))
                    if edge not in edges:
                        edges.append(edge)
        return self.num_vertices, edges
    
    def clique_expansion_low(self) -> Tuple[int, List[Tuple[int, int]]]:
        """clique_expansion的别名，保持向后兼容"""
        return self.clique_expansion()
    
    def is_connected(self, v1: int, v2: int) -> bool:
        """
        检查两个顶点是否在同一超边中
        
        Args:
            v1, v2: 顶点ID
            
        Returns:
            是否连通
        """
        for edge in self.e[0]:
            if v1 in edge and v2 in edge:
                return True
        return False
    
    def shortest_path_bfs(self, source: int, target: int) -> Optional[List[int]]:
        """
        使用BFS查找最短路径
        
        Args:
            source: 源顶点
            target: 目标顶点
            
        Returns:
            最短路径（顶点列表），如果不可达则返回None
        """
        if source == target:
            return [source]
        
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.neighbors(current):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def is_reachable(self, source: int, target: int) -> bool:
        """
        检查从source到target是否可达
        
        Args:
            source: 源顶点
            target: 目标顶点
            
        Returns:
            是否可达
        """
        return self.shortest_path_bfs(source, target) is not None
    
    def get_connected_component(self, vertex: int) -> Set[int]:
        """
        获取包含指定顶点的连通分量
        
        Args:
            vertex: 顶点ID
            
        Returns:
            连通分量中的所有顶点
        """
        component = {vertex}
        queue = [vertex]
        
        while queue:
            current = queue.pop(0)
            neighbors = self.neighbors(current)
            
            for neighbor in neighbors:
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        
        return component
    
    def shared_vertices(self, edge_idx1: int, edge_idx2: int) -> Set[int]:
        """
        获取两条超边之间的共享顶点
        
        Args:
            edge_idx1, edge_idx2: 超边索引
            
        Returns:
            共享顶点集合
        """
        if 0 <= edge_idx1 < len(self.e[0]) and 0 <= edge_idx2 < len(self.e[0]):
            edge1 = set(self.e[0][edge_idx1])
            edge2 = set(self.e[0][edge_idx2])
            return edge1 & edge2
        return set()
    
    def vertex_set_in_hyperedge(self, vertex_set: Set[int], edge_idx: int) -> bool:
        """
        检查顶点集是否全部在指定超边中
        
        Args:
            vertex_set: 顶点集合
            edge_idx: 超边索引
            
        Returns:
            是否全部包含
        """
        if 0 <= edge_idx < len(self.e[0]):
            edge = set(self.e[0][edge_idx])
            return vertex_set.issubset(edge)
        return False
    
    def shuffle_nodes(self) -> 'HyperGraph':
        """
        随机打乱顶点编号（用于同构测试）
        
        Returns:
            新的超图实例
        """
        verts = copy.deepcopy(self.v)
        random.shuffle(verts)
        
        new_hyperedges = []
        for edge in self.e[0]:
            new_edge = tuple(sorted([verts[v] for v in edge]))
            new_hyperedges.append(new_edge)
        
        return HyperGraph(verts, new_hyperedges)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'vertices': self.v,
            'hyperedges': self.e[0],
            'num_vertices': self.num_vertices,
            'num_edges': self.num_edges,
        }
    
    def get_vertices_with_degree(self, target_degree: int) -> List[int]:
        """
        获取特定degree的所有顶点
        
        Args:
            target_degree: 目标degree值
            
        Returns:
            满足条件的顶点列表
        """
        result = []
        for v in self.v:
            if self.degree(v) == target_degree:
                result.append(v)
        return sorted(result)
    
    def get_hyperedges_with_order(self, target_order: int) -> List[int]:
        """
        获取特定order的所有超边索引
        
        Args:
            target_order: 目标order值（顶点个数）
            
        Returns:
            满足条件的超边索引列表
        """
        result = []
        for i, edge in enumerate(self.e[0]):
            if len(edge) == target_order:
                result.append(i)
        return result
    
    def get_n_neighbors(self, vertex: int, min_order: int) -> Set[int]:
        """
        获取顶点的n-neighbor（仅考虑order>=min_order的超边）
        
        Args:
            vertex: 目标顶点
            min_order: 最小超边order
            
        Returns:
            满足条件的邻居顶点集合
        """
        neighbors = set()
        for edge in self.e[0]:
            if vertex in edge and len(edge) >= min_order:
                neighbors.update(set(edge) - {vertex})
        return neighbors
    
    def order_based_shortest_path(self, source: int, target: int) -> Tuple[Optional[List[int]], int]:
        """
        基于超边order为权重的最短路径（Dijkstra算法）
        
        Args:
            source: 源顶点
            target: 目标顶点
            
        Returns:
            (路径超边索引列表, 总权重)，若不可达则返回(None, float('inf'))
        """
        import heapq
        
        if source == target:
            return ([], 0)
        
        # dist[v] = (距离, 前驱顶点, 使用的超边索引)
        dist = {v: (float('inf'), None, None) for v in self.v}
        dist[source] = (0, None, None)
        
        # 优先队列：(当前距离, 当前顶点)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            curr_dist, curr_v = heapq.heappop(pq)
            
            if curr_v in visited:
                continue
            visited.add(curr_v)
            
            if curr_v == target:
                # 重构路径
                path_edges = []
                v = target
                while dist[v][2] is not None:
                    path_edges.append(dist[v][2])
                    v = dist[v][1]
                return (list(reversed(path_edges)), curr_dist)
            
            # 遍历包含当前顶点的所有超边
            for edge_idx, edge in enumerate(self.e[0]):
                if curr_v in edge:
                    edge_weight = len(edge)  # order作为权重
                    for next_v in edge:
                        if next_v != curr_v and next_v not in visited:
                            new_dist = curr_dist + edge_weight
                            if new_dist < dist[next_v][0]:
                                dist[next_v] = (new_dist, curr_v, edge_idx)
                                heapq.heappush(pq, (new_dist, next_v))
        
        return (None, float('inf'))
    
    def find_maximum_hypercycle(self) -> Tuple[int, Optional[List[int]]]:
        """
        找到最大严格Hypercycle（包含超边数最多的严格环）
        
        严格Hypercycle定义：超边序列 e1,e2,...,ek 其中：
        - 所有超边互不相同
        - 相邻超边恰好共享1个顶点（ei ∩ ei+1 = 1个顶点）
        - ek ∩ e1 = 1个顶点（形成闭环）
        
        Returns:
            (超环超边数, 超边索引列表)，若无超环返回(0, None)
        """
        max_cycle_length = 0
        max_cycle_edges = None
        
        def dfs_cycle(path_edges: List[int], last_shared_vertex: Optional[int]):
            """
            DFS搜索严格Hypercycle
            
            Args:
                path_edges: 当前路径上的超边索引列表
                last_shared_vertex: 上一条超边与当前超边共享的顶点
            """
            nonlocal max_cycle_length, max_cycle_edges
            
            if len(path_edges) == 0:
                return
            
            current_edge_idx = path_edges[-1]
            current_edge_set = set(self.e[0][current_edge_idx])
            
            # 尝试添加下一条超边
            for next_edge_idx, next_edge in enumerate(self.e[0]):
                if next_edge_idx in path_edges:  # 超边已在路径中
                    continue
                
                next_edge_set = set(next_edge)
                shared_vertices = current_edge_set & next_edge_set
                
                # 必须恰好共享1个顶点
                if len(shared_vertices) != 1:
                    continue
                
                shared_v = list(shared_vertices)[0]
                
                # 检查是否能回到起点形成环
                if len(path_edges) >= 2:  # 至少3条边才能形成环
                    first_edge_set = set(self.e[0][path_edges[0]])
                    first_shared = next_edge_set & first_edge_set
                    
                    if len(first_shared) == 1:
                        # 形成环！
                        cycle_length = len(path_edges) + 1
                        if cycle_length > max_cycle_length:
                            max_cycle_length = cycle_length
                            max_cycle_edges = path_edges + [next_edge_idx]
                
                # 继续搜索
                dfs_cycle(path_edges + [next_edge_idx], shared_v)
        
        # 从每条超边开始尝试构建环
        for start_edge_idx in range(len(self.e[0])):
            dfs_cycle([start_edge_idx], None)
        
        return (max_cycle_length, max_cycle_edges)
    
    def maximum_matching(self) -> int:
        """
        计算超图的最大匹配数（贪心近似算法）
        
        Returns:
            最大匹配的超边数量
        """
        # 贪心策略：优先选择顶点数少的超边
        sorted_edges = sorted(enumerate(self.e[0]), key=lambda x: len(x[1]))
        
        used_vertices = set()
        matching_count = 0
        
        for edge_idx, edge in sorted_edges:
            # 检查是否与已选超边共享顶点
            if not (set(edge) & used_vertices):
                used_vertices.update(edge)
                matching_count += 1
        
        return matching_count
    
    def minimum_vertex_cover(self) -> int:
        """
        计算最小顶点覆盖数（精确算法：分支定界+剪枝）
        
        Returns:
            最小覆盖的顶点数量
        """
        if not self.e[0]:
            return 0
        
        # 预先计算每个顶点覆盖的超边集合
        vertex_edges = {v: set() for v in self.v}
        for edge_idx, edge in enumerate(self.e[0]):
            for v in edge:
                vertex_edges[v].add(edge_idx)
        
        # 先用贪心算法得到上界
        upper_bound = self._greedy_vertex_cover()
        
        # 分支定界搜索
        min_cover_size = [upper_bound]  # 使用列表以便在递归中修改
        
        def branch_and_bound(uncovered_edges, current_cover, vertices_left):
            """
            分支定界递归搜索
            
            Args:
                uncovered_edges: 未覆盖的超边集合
                current_cover: 当前覆盖集合
                vertices_left: 剩余可选顶点列表
            """
            # 剪枝1：如果当前覆盖已经不优于已知最优解，返回
            if len(current_cover) >= min_cover_size[0]:
                return
            
            # 终止条件：所有超边已被覆盖
            if not uncovered_edges:
                min_cover_size[0] = len(current_cover)
                return
            
            # 剪枝2：下界估计 - 如果剩余超边数>剩余空间，不可能找到更优解
            if len(uncovered_edges) > min_cover_size[0] - len(current_cover):
                # 选择一条未覆盖的超边，至少要选其中一个顶点
                sample_edge = self.e[0][next(iter(uncovered_edges))]
                if len(current_cover) + 1 >= min_cover_size[0]:
                    return
            
            if not vertices_left:
                return
            
            # 选择覆盖最多未覆盖超边的顶点（启发式排序）
            best_vertex = max(vertices_left, 
                            key=lambda v: len(vertex_edges[v] & uncovered_edges))
            
            # 分支1：选择这个顶点
            new_uncovered = uncovered_edges - vertex_edges[best_vertex]
            new_vertices = [v for v in vertices_left if v != best_vertex]
            branch_and_bound(new_uncovered, current_cover | {best_vertex}, new_vertices)
            
            # 分支2：不选择这个顶点（但要确保覆盖它参与的超边）
            branch_and_bound(uncovered_edges, current_cover, new_vertices)
        
        all_edges = set(range(len(self.e[0])))
        vertices_list = sorted(self.v, key=lambda v: len(vertex_edges[v]), reverse=True)
        
        branch_and_bound(all_edges, set(), vertices_list)
        
        return min_cover_size[0]
    
    def _greedy_vertex_cover(self) -> int:
        """
        贪心算法计算顶点覆盖（用作分支定界的上界）
        
        Returns:
            贪心覆盖的顶点数量
        """
        uncovered_edges = set(range(len(self.e[0])))
        cover = set()
        
        while uncovered_edges:
            # 找到覆盖最多未覆盖超边的顶点
            best_vertex = None
            best_coverage = 0
            
            for v in self.v:
                if v in cover:
                    continue
                coverage = sum(1 for edge_idx in uncovered_edges 
                             if v in self.e[0][edge_idx])
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_vertex = v
            
            if best_vertex is None:
                break
            
            cover.add(best_vertex)
            uncovered_edges = {idx for idx in uncovered_edges 
                             if best_vertex not in self.e[0][idx]}
        
        return len(cover)
    
    def has_hamiltonian_path(self, source: int, target: int) -> Tuple[bool, Optional[List[int]]]:
        """
        检查是否存在哈密顿路径（经过所有顶点恰好一次）
        
        Args:
            source: 起点
            target: 终点
            
        Returns:
            (是否存在, 超边索引路径)
        """
        all_vertices = set(self.v)
        result_path = []
        
        def dfs_hamiltonian(curr_v: int, visited: Set[int], path_edges: List[int]) -> bool:
            # 所有顶点已访问
            if len(visited) == len(all_vertices):
                if curr_v == target:
                    result_path.extend(path_edges)
                    return True
                return False
            
            # 尝试通过每条包含当前顶点的超边扩展
            for edge_idx, edge in enumerate(self.e[0]):
                if curr_v not in edge:
                    continue
                
                # 找到未访问的邻居
                next_vertices = set(edge) - visited
                
                for next_v in next_vertices:
                    if dfs_hamiltonian(next_v, visited | {next_v}, path_edges + [edge_idx]):
                        return True
            
            return False
        
        if dfs_hamiltonian(source, {source}, []):
            return (True, result_path)
        return (False, None)
    
    @staticmethod
    def from_dict(data: Dict) -> 'HyperGraph':
        """从字典创建超图"""
        return HyperGraph(data['vertices'], data['hyperedges'])


def generate_random_hypergraph(
    num_vertices: int,
    num_edges: int,
    edge_size_range: Tuple[int, int] = (2, 5),
    seed: Optional[int] = None
) -> HyperGraph:
    """
    生成随机超图
    
    Args:
        num_vertices: 顶点数量
        num_edges: 超边数量
        edge_size_range: 超边大小范围 (min, max)
        seed: 随机种子
        
    Returns:
        HyperGraph实例
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    vertices = list(range(num_vertices))
    hyperedges = []
    
    min_size, max_size = edge_size_range
    
    for _ in range(num_edges):
        # 随机选择超边大小
        edge_size = random.randint(min_size, min(max_size, num_vertices))
        
        # 随机选择顶点
        edge = tuple(sorted(random.sample(vertices, edge_size)))
        
        # 避免重复超边
        if edge not in hyperedges:
            hyperedges.append(edge)
    
    return HyperGraph(vertices, hyperedges)


def generate_connected_hypergraph(
    num_vertices: int,
    num_edges: int,
    edge_size_range: Tuple[int, int] = (2, 5),
    seed: Optional[int] = None
) -> HyperGraph:
    """
    生成连通的随机超图
    
    Args:
        num_vertices: 顶点数量
        num_edges: 超边数量
        edge_size_range: 超边大小范围
        seed: 随机种子
        
    Returns:
        连通的HyperGraph实例
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    vertices = list(range(num_vertices))
    hyperedges = []
    
    # 首先确保连通性：创建一条包含所有顶点的路径
    min_size, max_size = edge_size_range
    
    # 随机打乱顶点
    shuffled = vertices.copy()
    random.shuffle(shuffled)
    
    # 创建初始超边以确保连通
    i = 0
    while i < num_vertices:
        remaining = num_vertices - i
        if remaining < min_size:
            # 剩余顶点不足，全部加入最后一个超边
            if remaining >= 2:
                edge = tuple(sorted(shuffled[i:]))
                hyperedges.append(edge)
            break
        
        edge_size = random.randint(min_size, min(max_size, remaining))
        edge = tuple(sorted(shuffled[i:i + edge_size]))
        if len(edge) >= 2:
            hyperedges.append(edge)
        i += max(1, edge_size - 1)  # 重叠一个顶点以保证连通，至少前进1步
    
    # 添加额外的随机超边
    while len(hyperedges) < num_edges:
        edge_size = random.randint(min_size, min(max_size, num_vertices))
        edge = tuple(sorted(random.sample(vertices, edge_size)))
        
        if edge not in hyperedges:
            hyperedges.append(edge)
    
    return HyperGraph(vertices, hyperedges)


def load_hypergraph_from_file(filepath: str) -> HyperGraph:
    """
    从文件加载超图
    
    Args:
        filepath: 文件路径
        
    Returns:
        HyperGraph实例
    """
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, HyperGraph):
        return data
    elif isinstance(data, dict):
        return HyperGraph.from_dict(data)
    else:
        raise ValueError(f"Unknown hypergraph format in {filepath}")


def save_hypergraph_to_file(hypergraph: HyperGraph, filepath: str):
    """
    保存超图到文件
    
    Args:
        hypergraph: HyperGraph实例
        filepath: 文件路径
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(hypergraph, f)
