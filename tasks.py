"""
VLMBench Task Definitions - Refactored with Unified Answer Checking

定义所有超图理解与推理任务
基础理解任务（6个）：聚焦超图核心结构与阶特性
高阶推理任务（6个）：基于order的算法级推理

统一的answer checking方法：
1. 数值类任务：统一使用 check_numeric_answer
2. Yes/No类任务：统一使用 check_binary_answer
3. 集合类任务：统一使用 check_set_answer
4. 路径验证类：使用 check_path_answer 进行完整性验证
"""

import random
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from hypergraph_utils import HyperGraph
from text_encoder import encode_graph


# ========== 统一的答案检查辅助函数 ==========

def extract_answer_part(predicted: str) -> str:
    """
    统一提取"Ans:"后的部分，避免匹配问题描述或推理过程中的内容。
    
    Args:
        predicted: 预测的完整回答
        
    Returns:
        "Ans:"后的部分（小写）
    """
    predicted_lower = str(predicted).lower()
    ans_match = re.search(r'ans[:\s]+(.*)', predicted_lower, re.DOTALL)
    if ans_match:
        return ans_match.group(1).strip()
    return predicted_lower


def parse_ground_truth_numeric(ground_truth: Any) -> Optional[int]:
    """
    从ground_truth中提取数值（支持字符串和整数格式）。
    
    Args:
        ground_truth: 可能是整数、字符串"Ans:8."或None
        
    Returns:
        提取的整数值，或None（表示无答案）
    """
    if ground_truth is None:
        return None
    
    if isinstance(ground_truth, (int, float)):
        return int(ground_truth)
    
    # 字符串格式
    gt_str = str(ground_truth).lower()
    
    # 检查是否表示"无答案"
    if any(keyword in gt_str for keyword in ['no path', 'no answer', 'none', 'n/a']):
        return None
    
    # 提取数字
    nums = re.findall(r'\d+', gt_str)
    return int(nums[0]) if nums else None


def parse_ground_truth_binary(ground_truth: Any) -> str:
    """
    从ground_truth中提取Yes/No答案。
    
    Args:
        ground_truth: 可能是"Yes"/"No"或字符串"Ans:Yes."
        
    Returns:
        'yes' 或 'no'
    """
    gt_str = str(ground_truth).lower()
    return 'yes' if 'yes' in gt_str else 'no'


def check_numeric_answer(predicted: str, ground_truth: Any) -> bool:
    """
    统一的数值答案检查方法。
    
    适用任务：
    - VertexCount, HyperedgeCount
    - SpecificVertexDegreeCount, SpecificOrderHyperedgeCount
    - OrderBasedShortestPath (路径长度)
    - OrderBasedMaxFlow
    - HypergraphVertexCover
    
    Args:
        predicted: 预测答案
        ground_truth: 正确答案（整数或字符串）
        
    Returns:
        是否正确
    """
    predicted_part = extract_answer_part(predicted)
    gt_value = parse_ground_truth_numeric(ground_truth)
    
    # 检查"无答案"情况
    if any(keyword in predicted_part for keyword in ['no path', 'no answer', 'none', 'n/a']):
        return gt_value is None
    
    # ground_truth是None但预测了数值
    if gt_value is None:
        return False
    
    # 提取预测的数值（使用第一个数字）
    pred_nums = re.findall(r'\d+', predicted_part)
    if not pred_nums:
        return False
    
    return int(pred_nums[0]) == gt_value


def check_binary_answer(predicted: str, ground_truth: Any) -> bool:
    """
    统一的Yes/No答案检查方法。
    
    适用任务：
    - StrictHypercycle
    - HypergraphIsomorphism
    
    Args:
        predicted: 预测答案
        ground_truth: 正确答案（"Yes"/"No"或字符串）
        
    Returns:
        是否正确
    """
    predicted_part = extract_answer_part(predicted)
    gt_answer = parse_ground_truth_binary(ground_truth)
    
    # 检查是否包含yes或no（使用词边界避免误匹配）
    has_yes = bool(re.search(r'\byes\b', predicted_part))
    has_no = bool(re.search(r'\bno\b', predicted_part))
    
    # 只有明确的yes（无no）或no（无yes）才判定
    if has_yes and not has_no:
        return gt_answer == 'yes'
    elif has_no and not has_yes:
        return gt_answer == 'no'
    
    return False


def check_set_answer(predicted: str, ground_truth: Any) -> bool:
    """
    统一的集合答案检查方法。
    
    适用任务：
    - DirectNeighbor
    - NNeighbor
    
    Args:
        predicted: 预测答案
        ground_truth: 正确答案（顶点ID列表或字符串）
        
    Returns:
        是否正确
    """
    predicted_part = extract_answer_part(predicted)
    
    # 处理"无邻居"的情况
    if re.search(r'\bno\s+(neighbors?|n-neighbors?)\b', predicted_part):
        if isinstance(ground_truth, str):
            gt_has_none = bool(re.search(r'\bno\s+neighbors?\b', ground_truth.lower()))
            gt_vertices = re.findall(r'v_?\{?(\d+)\}?', ground_truth)
            return gt_has_none or len(gt_vertices) == 0
        return len(ground_truth) == 0
    
    # 提取预测的顶点编号 (支持v0, v_0, v_{0}等格式)
    pred_vertices = set(re.findall(r'v_?\{?(\d+)\}?', predicted_part))
    
    # 如果没有提取到顶点
    if not pred_vertices:
        if isinstance(ground_truth, str):
            return len(re.findall(r'v_?\{?(\d+)\}?', ground_truth)) == 0
        return len(ground_truth) == 0
    
    # 从ground_truth中提取顶点编号
    if isinstance(ground_truth, str):
        truth_vertices = set(re.findall(r'v_?\{?(\d+)\}?', ground_truth))
    else:
        truth_vertices = set(str(v) for v in ground_truth)
    
    # 精确匹配
    return pred_vertices == truth_vertices


# ========== 基类定义 ==========

class BaseTask:
    """任务基类"""
    
    task_name = "BaseTask"
    category = "base"
    difficulty = 0
    description = "Base task class"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        """
        准备一个任务样本
        
        Args:
            hypergraph: 超图对象
            encoding_method: 文本编码方式
            vqa_mode: VQA模式
            
        Returns:
            样本字典
        """
        raise NotImplementedError
    
    @classmethod
    def generate_question_text(
        cls,
        encoded_text: str,
        **kwargs
    ) -> str:
        """
        生成问题文本（包含图描述）
        
        Args:
            encoded_text: 编码后的超图文本
            **kwargs: 额外参数
            
        Returns:
            问题文本（包含图描述）
        """
        raise NotImplementedError
    
    @classmethod
    def get_question_only(cls, **kwargs) -> str:
        """
        获取纯问题文本（不包含图描述）
        用于Vision-Only模式
        
        Args:
            **kwargs: 任务特定参数
            
        Returns:
            纯问题文本
        """
        raise NotImplementedError
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        """
        检查答案是否正确
        
        Args:
            predicted: 预测答案
            ground_truth: 正确答案
            
        Returns:
            是否正确
        """
        raise NotImplementedError


# ========== 基础理解任务（6个）==========

class VertexCount(BaseTask):
    """任务1：顶点数量统计"""
    
    task_name = "VertexCount"
    category = "structure_understanding"
    difficulty = "easy"
    description = "Count the total number of vertices in the hypergraph"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        answer = hypergraph.num_vertices
        
        question_text = cls.generate_question_text(encoded_text)
        question_only = cls.get_question_only()
        
        return {
            'question_text': question_text,
            'question_only': question_only,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer,
            'hypergraph': hypergraph,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        return (
            f"{encoded_text}\n"
            f"Q: How many vertices are in the hypergraph G? "
            f"List the answer after 'Ans:'.\nA: "
        )
    
    @classmethod
    def get_question_only(cls, **kwargs) -> str:
        return "How many vertices are in the hypergraph G? List the answer after 'Ans:'."
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_numeric_answer(predicted, ground_truth)


class HyperedgeCount(BaseTask):
    """任务2：超边数量统计"""
    
    task_name = "HyperedgeCount"
    category = "structure_understanding"
    difficulty = "easy"
    description = "Count the total number of hyperedges in the hypergraph"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        answer = hypergraph.num_edges
        
        question_text = cls.generate_question_text(encoded_text)
        question_only = cls.get_question_only()
        
        return {
            'question_text': question_text,
            'question_only': question_only,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer,
            'hypergraph': hypergraph,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        return (
            f"{encoded_text}\n"
            f"Q: How many hyperedges are in the hypergraph G? "
            f"List the answer after 'Ans:'.\nA: "
        )
    
    @classmethod
    def get_question_only(cls, **kwargs) -> str:
        return "How many hyperedges are in the hypergraph G? List the answer after 'Ans:'."
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_numeric_answer(predicted, ground_truth)


class SpecificVertexDegreeCount(BaseTask):
    """任务3：特定degree的顶点数统计"""
    
    task_name = "SpecificVertexDegreeCount"
    category = "structure_understanding"
    difficulty = "medium"
    description = "Count vertices with a specific degree (number of incident hyperedges)"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 从图中已存在的unique degrees中随机选择
        all_degrees = [hypergraph.degree(v) for v in hypergraph.v]
        unique_degrees = list(set(all_degrees))
        target_degree = random.choice(unique_degrees) if unique_degrees else 0
        
        vertices_with_degree = hypergraph.get_vertices_with_degree(target_degree)
        answer = len(vertices_with_degree)
        
        question_text = cls.generate_question_text(encoded_text, target_degree=target_degree)
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer,
            'hypergraph': hypergraph,
            'target_degree': target_degree,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        target_degree = kwargs.get('target_degree', 0)
        return (
            f"{encoded_text}\n"
            f"Q: How many vertices have degree {target_degree} in hypergraph G? "
            f"(Degree = number of hyperedges the vertex belongs to). "
            f"List the answer after 'Ans:'.\nA: "
        )
    
    @classmethod
    def get_question_only(cls, **kwargs) -> str:
        target_degree = kwargs.get('target_degree', 0)
        return (f"How many vertices have degree {target_degree} in hypergraph G? "
                f"(Degree = number of hyperedges the vertex belongs to). "
                f"List the answer after 'Ans:'.")
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_numeric_answer(predicted, ground_truth)


class SpecificOrderHyperedgeCount(BaseTask):
    """任务4：特定order的超边数统计"""
    
    task_name = "SpecificOrderHyperedgeCount"
    category = "structure_understanding"
    difficulty = "medium"
    description = "Count hyperedges with a specific order (number of vertices)"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 从图中已存在的unique orders中随机选择
        all_orders = [len(edge) for edge in hypergraph.e[0]]
        unique_orders = list(set(all_orders))
        target_order = random.choice(unique_orders) if unique_orders else 2
        
        edges_with_order = hypergraph.get_hyperedges_with_order(target_order)
        answer = len(edges_with_order)
        
        question_text = cls.generate_question_text(encoded_text, target_order=target_order)
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer,
            'hypergraph': hypergraph,
            'target_order': target_order,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        target_order = kwargs.get('target_order', 2)
        return (
            f"{encoded_text}\n"
            f"Q: How many hyperedges have order {target_order} in hypergraph G? "
            f"(Order = number of vertices in the hyperedge). "
            f"List the answer after 'Ans:'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_numeric_answer(predicted, ground_truth)


class DirectNeighbor(BaseTask):
    """任务5：直接邻居识别"""
    
    task_name = "DirectNeighbor"
    category = "structure_understanding"
    difficulty = "easy"
    description = "Identify all direct neighbors of a given vertex"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 随机选择一个顶点
        target_vertex = random.choice(hypergraph.v)
        
        neighbors = hypergraph.neighbors(target_vertex)
        answer_set = sorted(neighbors)
        
        if answer_set:
            answer = "{" + ",".join(f"v{v}" for v in answer_set) + "}"
        else:
            answer = "No neighbors"
        
        question_text = cls.generate_question_text(encoded_text, target_vertex=target_vertex)
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer_set,
            'hypergraph': hypergraph,
            'target_vertex': target_vertex,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        target_vertex = kwargs.get('target_vertex', 0)
        return (
            f"{encoded_text}\n"
            f"Q: What are the direct neighbors of vertex v{target_vertex} in hypergraph G? "
            f"(Neighbors = vertices sharing at least one hyperedge with v{target_vertex}). "
            f"List the answer after 'Ans:' in the format {{v1,v2,...}} or 'No neighbors'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_set_answer(predicted, ground_truth)


class NNeighbor(BaseTask):
    """任务6：order>=n约束下的邻居识别"""
    
    task_name = "NNeighbor"
    category = "structure_understanding"
    difficulty = "medium"
    description = "Identify neighbors considering only hyperedges with order >= n"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 随机选择一个顶点和最小order
        target_vertex = random.choice(hypergraph.v)
        all_orders = [len(edge) for edge in hypergraph.e[0]]
        min_order = random.choice(all_orders) if all_orders else 2
        
        neighbors = hypergraph.get_n_neighbors(target_vertex, min_order)
        answer_set = sorted(neighbors)
        
        if answer_set:
            answer = "{" + ",".join(f"v{v}" for v in answer_set) + "}"
        else:
            answer = "No n-neighbors"
        
        question_text = cls.generate_question_text(
            encoded_text, 
            target_vertex=target_vertex,
            min_order=min_order
        )
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer_set,
            'hypergraph': hypergraph,
            'target_vertex': target_vertex,
            'min_order': min_order,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        target_vertex = kwargs.get('target_vertex', 0)
        min_order = kwargs.get('min_order', 2)
        return (
            f"{encoded_text}\n"
            f"Q: What are the neighbors of vertex v{target_vertex} when only considering "
            f"hyperedges with order >= {min_order} in hypergraph G? "
            f"List the answer after 'Ans:' in the format {{v1,v2,...}} or 'No n-neighbors'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_set_answer(predicted, ground_truth)


# ========== 高阶推理任务（6个）==========

class OrderBasedShortestPath(BaseTask):
    """任务7：基于超边order的最短路径长度"""
    
    task_name = "OrderBasedShortestPath"
    category = "algorithmic_reasoning"
    difficulty = "medium"
    description = "Find shortest path length using hyperedge order as edge weight"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 尝试选择连通的source-target对
        if len(hypergraph.v) < 2:
            source, target = 0, 0
        else:
            max_attempts = 10
            source, target = None, None
            for _ in range(max_attempts):
                s, t = random.sample(hypergraph.v, 2)
                if hypergraph.is_reachable(s, t):
                    source, target = s, t
                    break
            
            if source is None:
                source, target = random.sample(hypergraph.v, 2)
        
        path_edges, total_weight = hypergraph.order_based_shortest_path(source, target)
        
        # 只返回权重
        if path_edges is not None:
            answer = total_weight
        else:
            answer = "No path"
        
        question_text = cls.generate_question_text(
            encoded_text,
            source=source,
            target=target
        )
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': total_weight if path_edges is not None else None,
            'hypergraph': hypergraph,
            'source': source,
            'target': target,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        source = kwargs.get('source', 0)
        target = kwargs.get('target', 1)
        return (
            f"{encoded_text}\n"
            f"Q: What is the shortest path length from vertex v{source} to vertex v{target} in hypergraph G, "
            f"where each hyperedge's weight equals its order (number of vertices)? "
            f"If no path exists, answer 'No path'. List the answer after 'Ans:'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_numeric_answer(predicted, ground_truth)


class OrderBasedMaxFlow(BaseTask):
    """任务8：基于超边order的最大流"""
    
    task_name = "OrderBasedMaxFlow"
    category = "algorithmic_reasoning"
    difficulty = "hard"
    description = "Compute maximum flow using hyperedge order as capacity"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 随机选择源和汇顶点
        if len(hypergraph.v) < 2:
            source, sink = 0, 0
        else:
            source, sink = random.sample(hypergraph.v, 2)
        
        max_flow = cls._estimate_max_flow(hypergraph, source, sink)
        answer = max_flow
        
        question_text = cls.generate_question_text(
            encoded_text,
            source=source,
            sink=sink
        )
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer,
            'hypergraph': hypergraph,
            'source': source,
            'sink': sink,
        }
    
    @classmethod
    def _estimate_max_flow(cls, hg: HyperGraph, source: int, sink: int) -> int:
        """
        计算最大流：将超图转化为流网络
        模型：source -> 顶点 -> 超边 -> 顶点 -> sink
        - source到包含source的超边中的其他顶点：容量无限
        - 顶点到包含它的超边：容量 = 超边的order
        - 超边到包含sink的顶点：容量 = 超边的order
        """
        from collections import defaultdict, deque
        
        if not hg.is_reachable(source, sink):
            return 0
        
        # 构建流网络
        INF = 10000  # 用大数代替无限大
        capacity = defaultdict(lambda: defaultdict(int))
        
        # 为每条超边创建一个中间节点
        for edge_idx, edge in enumerate(hg.e[0]):
            edge_node = f'e{edge_idx}'
            edge_capacity = len(edge)
            
            for v in edge:
                # 顶点 -> 超边：容量 = 超边order
                capacity[v][edge_node] = max(capacity[v][edge_node], edge_capacity)
                
                # 超边 -> 顶点：容量 = 超边order
                capacity[edge_node][v] = max(capacity[edge_node][v], edge_capacity)
        
        # BFS找增广路径
        def bfs(source, sink, parent):
            visited = {source}
            queue = deque([source])
            
            while queue:
                u = queue.popleft()
                
                for v in capacity[u]:
                    if v not in visited and capacity[u][v] > 0:
                        visited.add(v)
                        parent[v] = u
                        if v == sink:
                            return True
                        queue.append(v)
            return False
        
        # Edmonds-Karp算法
        max_flow_value = 0
        parent = {}
        
        while bfs(source, sink, parent):
            # 找到增广路径的最小容量
            path_flow = INF
            s = sink
            while s != source:
                path_flow = min(path_flow, capacity[parent[s]][s])
                s = parent[s]
            
            # 更新残余网络
            v = sink
            while v != source:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = parent[v]
            
            max_flow_value += path_flow
            parent = {}
        
        return int(max_flow_value)
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        source = kwargs.get('source', 0)
        sink = kwargs.get('sink', 1)
        return (
            f"{encoded_text}\n"
            f"Q: What is the estimated maximum flow from vertex v{source} to vertex v{sink} "
            f"in hypergraph G, where each hyperedge's capacity equals its order? "
            f"If no flow exists, answer '0'. List the answer after 'Ans:'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_numeric_answer(predicted, ground_truth)


class StrictHypercycle(BaseTask):
    """任务9：严格Hypercycle规划与验证"""
    
    task_name = "StrictHypercycle"
    category = "algorithmic_reasoning"
    difficulty = "medium"
    description = "Find a strict hypercycle in the hypergraph and validate it (adjacent hyperedges share exactly one vertex)"
    use_guaranteed_solution = True  # 标记：生成时使用保证有解的构造方法
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal',
        guaranteed_cycle: List[int] = None
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 如果提供了保证的cycle，直接使用；否则尝试查找
        if guaranteed_cycle is not None:
            cycle_edges = guaranteed_cycle
            has_cycle = True
        else:
            max_vertices, cycle_edges = hypergraph.find_maximum_hypercycle()
            has_cycle = max_vertices > 0 and cycle_edges is not None
        
        if has_cycle and cycle_edges:
            answer = f"Yes,Cycle:[{','.join(f'e{i}' for i in cycle_edges)}]"
        else:
            answer = "No"
        
        question_text = cls.generate_question_text(encoded_text)
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': {
                'has_cycle': has_cycle,
                'cycle_edges': cycle_edges,
                'hyperedges': [list(edge) for edge in hypergraph.e[0]],
            },
            'metadata': {
                'cycle_edges': cycle_edges,
                'has_cycle': has_cycle,
            },
            'has_cycle': has_cycle,
            'hypergraph': hypergraph,  # 用于验证，但不会被序列化到JSON
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        return (
            f"{encoded_text}\n"
            f"Q: Does hypergraph G contain a strict hypercycle? "
            f"(A strict hypercycle is a sequence of hyperedges e1,e2,...,ek where adjacent hyperedges "
            f"share exactly one vertex, i.e., |ei ∩ ei+1| = 1, and |ek ∩ e1| = 1, forming a closed loop). "
            f"If yes, provide the cycle. List the answer after 'Ans:' as 'Yes,Cycle:[e0,e1,...]' or 'No'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        """
        验证strict hypercycle答案（Planning + Validation）：
        1. 如果预测No，验证ground_truth确实不存在cycle
        2. 如果预测Yes，提取并验证cycle的有效性
        """
        predicted_part = extract_answer_part(predicted)
        
        has_yes = bool(re.search(r'\byes\b', predicted_part))
        has_no = bool(re.search(r'\bno\b', predicted_part))
        
        # 提取ground_truth信息
        if isinstance(ground_truth, dict):
            gt_has_cycle = ground_truth.get('has_cycle', False)
            gt_hyperedges = ground_truth.get('hyperedges', [])
        else:
            # 兼容旧格式
            gt_has_cycle = 'yes' in str(ground_truth).lower()
            gt_hyperedges = []
        
        # 预测No
        if has_no and not has_yes:
            return not gt_has_cycle
        
        # 预测Yes - 必须验证cycle
        if has_yes:
            if not gt_has_cycle:
                return False
            
            # 提取cycle: 支持多种格式
            # Cycle:[e0,e1,...] 或 Cycle: [e0, e1, ...] 或 cycle:[0,1,2] 等
            cycle_match = re.search(r'cycle\s*[:\-]?\s*\[([^\]]+)\]', predicted_part, re.IGNORECASE)
            if not cycle_match:
                # 尝试更宽松的匹配: cycle e0 e1 e2
                cycle_match = re.search(r'cycle\s*[:\-]?\s*(.+?)(?:\n|$)', predicted_part, re.IGNORECASE)
                if not cycle_match:
                    return False
            
            cycle_str = cycle_match.group(1).strip()
            try:
                # 解析edge索引 - 支持多种格式
                pred_cycle = []
                # 移除所有括号
                cycle_str = cycle_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                # 分割：支持逗号、空格、分号等分隔符
                items = re.split(r'[,;\s]+', cycle_str)
                for item in items:
                    item = item.strip()
                    if not item:
                        continue
                    # 提取数字：支持 e0, e_0, edge0, 0 等格式
                    match = re.search(r'(?:e(?:dge)?[_\-]?)?(\d+)', item, re.IGNORECASE)
                    if match:
                        edge_idx = int(match.group(1))
                        pred_cycle.append(edge_idx)
                
                if not pred_cycle or not gt_hyperedges:
                    return False
                
                # 验证cycle的有效性
                return cls._validate_cycle(pred_cycle, gt_hyperedges)
            except:
                return False
        
        return False
    
    @classmethod
    def _validate_cycle(cls, cycle_edges: list, hyperedges: list) -> bool:
        """
        验证给定的cycle是否有效
        要求：
        1. 所有edge索引有效
        2. 相邻edges共享恰好1个顶点
        3. 首尾edges也共享恰好1个顶点（形成闭环）
        """
        if len(cycle_edges) < 2:
            return False
        
        # 检查所有edge索引是否有效
        if any(idx < 0 or idx >= len(hyperedges) for idx in cycle_edges):
            return False
        
        # 检查相邻edges的交集
        for i in range(len(cycle_edges)):
            curr_idx = cycle_edges[i]
            next_idx = cycle_edges[(i + 1) % len(cycle_edges)]
            
            curr_edge = set(hyperedges[curr_idx])
            next_edge = set(hyperedges[next_idx])
            
            intersection = curr_edge & next_edge
            
            # 必须恰好共享1个顶点
            if len(intersection) != 1:
                return False
        
        return True


class HypergraphIsomorphism(BaseTask):
    """任务10：超图同构识别"""
    
    task_name = "HypergraphIsomorphism"
    category = "algorithmic_reasoning"
    difficulty = "hard"
    description = "Determine if two hypergraphs are isomorphic"
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal'
    ) -> Dict[str, Any]:
        # 50%概率生成同构的两个超图
        if random.random() > 0.5:
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
            is_isomorphic = True
        else:
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
            is_isomorphic = False
        
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
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': answer,
            'hypergraph': hypergraph,
            'hypergraph_H': hypergraph_H,
            'is_isomorphic': is_isomorphic,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        return encoded_text
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        return check_binary_answer(predicted, ground_truth)


class Hypergraph3Coloring(BaseTask):
    """任务11：超图3着色"""
    
    task_name = "Hypergraph3Coloring"
    category = "algorithmic_reasoning"
    difficulty = "hard"
    description = "Determine if hypergraph is 3-colorable (each hyperedge contains at least 2 colors)"
    use_guaranteed_solution = True  # 标记：生成时使用保证有解的构造方法
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal',
        guaranteed_coloring: dict = None
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 如果提供了保证的着色方案，直接使用；否则尝试查找
        if guaranteed_coloring is not None:
            is_colorable = True
            coloring = guaranteed_coloring
        else:
            is_colorable, coloring = cls._find_3_coloring(hypergraph)
        
        if is_colorable:
            answer = f"Yes,Coloring:[{','.join(f'v{v}:c{coloring[v]}' for v in sorted(coloring.keys()))}]"
        else:
            answer = "No"
        
        question_text = cls.generate_question_text(encoded_text)
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': {
                'is_colorable': is_colorable,
                'coloring': coloring,
                'hyperedges': [list(edge) for edge in hypergraph.e[0]],  # 保存超边用于验证
            },
            'metadata': {
                'is_colorable': is_colorable,
                'coloring': coloring,
            },
            'hypergraph': hypergraph,
        }
    
    @classmethod
    def _find_3_coloring(cls, hg: HyperGraph) -> Tuple[bool, Optional[dict]]:
        """
        尝试找到超图的3着色方案
        要求：每条超边必须包含至少2种不同颜色的顶点
        使用回溯搜索
        """
        num_vertices = len(hg.v)
        coloring = {}
        
        def is_valid_coloring():
            """检查当前着色是否满足约束"""
            for edge in hg.e[0]:
                edge_colors = set()
                for v in edge:
                    if v in coloring:
                        edge_colors.add(coloring[v])
                
                # 如果这条边的所有顶点都已着色，检查是否至少有2种颜色
                if len(edge_colors) == len(edge) and len(edge_colors) < 2:
                    return False
            return True
        
        def backtrack(vertex_idx):
            if vertex_idx == num_vertices:
                return is_valid_coloring()
            
            vertex = hg.v[vertex_idx]
            
            # 尝试3种颜色
            for color in [0, 1, 2]:
                coloring[vertex] = color
                
                # 检查当前着色是否有效
                valid = True
                for edge in hg.e[0]:
                    if vertex in edge:
                        # 检查这条边的着色情况
                        edge_colors = set()
                        all_colored = True
                        for v in edge:
                            if v in coloring:
                                edge_colors.add(coloring[v])
                            else:
                                all_colored = False
                        
                        # 如果边上所有顶点都着色了，且只有1种颜色，则无效
                        if all_colored and len(edge_colors) < 2:
                            valid = False
                            break
                
                if valid and backtrack(vertex_idx + 1):
                    return True
                
                del coloring[vertex]
            
            return False
        
        if backtrack(0):
            return True, coloring
        else:
            return False, None
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        return (
            f"{encoded_text}\n"
            f"Q: Can hypergraph G be 3-colored such that each hyperedge contains at least 2 different colors? "
            f"If yes, provide a valid coloring (assign each vertex a color from {{c0, c1, c2}}). "
            f"List the answer after 'Ans:' as 'Yes,Coloring:[v0:c0,v1:c1,...]' or 'No'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        """
        验证3着色答案：
        1. 如果预测No，验证确实无法3着色
        2. 如果预测Yes，必须提供有效的着色方案并验证
        """
        predicted_part = extract_answer_part(predicted)
        
        has_yes = bool(re.search(r'\byes\b', predicted_part))
        has_no = bool(re.search(r'\bno\b', predicted_part))
        
        # 提取ground_truth信息
        if isinstance(ground_truth, dict):
            gt_is_colorable = ground_truth.get('is_colorable', False)
            gt_coloring = ground_truth.get('coloring', {})
            gt_hyperedges = ground_truth.get('hyperedges', [])
            # JSON反序列化会把int key转成string，需要转回int
            if gt_coloring and isinstance(list(gt_coloring.keys())[0], str):
                gt_coloring = {int(k): v for k, v in gt_coloring.items()}
        else:
            gt_is_colorable = 'yes' in str(ground_truth).lower()
            gt_coloring = {}
            gt_hyperedges = []
        
        # 预测No
        if has_no and not has_yes:
            return not gt_is_colorable
        
        # 预测Yes - 必须验证着色方案
        if has_yes:
            if not gt_is_colorable:
                return False
            
            # 提取预测的着色方案 - 支持多种格式
            # Coloring:[v0:c0,v1:c1,...] 或 Coloring: {v0:c0, v1:c1} 等
            coloring_match = re.search(r'coloring\s*[:\-]?\s*[\[{]([^\]\}]+)[\]}]', predicted_part, re.IGNORECASE)
            if not coloring_match:
                # 尝试更宽松的匹配
                coloring_match = re.search(r'coloring\s*[:\-]?\s*(.+?)(?:\n|$)', predicted_part, re.IGNORECASE)
                if not coloring_match:
                    return False
            
            # 解析着色: v0:c0,v1:c1,... - 支持多种格式
            coloring_str = coloring_match.group(1).strip()
            # 移除括号
            coloring_str = coloring_str.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
            try:
                pred_coloring = {}
                # 分割：支持逗号、分号等
                items = re.split(r'[,;]', coloring_str)
                for item in items:
                    item = item.strip()
                    if not item:
                        continue
                    # 支持多种格式: v0:c0, v0=c0, vertex0:color0, 0:0 等
                    match = re.search(r'(?:v(?:ertex)?[_\-]?)?(\d+)\s*[:=]\s*(?:c(?:olor)?[_\-]?)?(\d+)', item, re.IGNORECASE)
                    if match:
                        vertex = int(match.group(1))
                        color = int(match.group(2))
                        if color not in [0, 1, 2]:
                            return False
                        pred_coloring[vertex] = color
                
                if not pred_coloring or not gt_coloring or not gt_hyperedges:
                    return False
                
                # 验证着色方案的有效性：使用ground_truth的超边信息
                return cls._validate_coloring_with_gt(gt_coloring, gt_hyperedges, pred_coloring)
            except:
                return False
        
        return False
    
    @classmethod
    def _validate_coloring_with_gt(cls, gt_coloring: dict, gt_hyperedges: list, pred_coloring: dict) -> bool:
        """
        使用ground_truth信息验证预测的着色方案
        
        Args:
            gt_coloring: ground_truth中的着色方案（用于确定顶点集合）
            gt_hyperedges: ground_truth中的超边列表
            pred_coloring: 预测的着色方案
            
        Returns:
            bool: 着色方案是否有效
        """
        # 检查所有顶点是否都着色了
        if set(pred_coloring.keys()) != set(gt_coloring.keys()):
            return False
        
        # 检查每条超边是否包含至少2种颜色
        for edge in gt_hyperedges:
            edge_colors = set(pred_coloring[v] for v in edge)
            if len(edge_colors) < 2:
                return False
        
        return True
    
    @classmethod
    def _validate_coloring(cls, hg: HyperGraph, coloring: dict) -> bool:
        """
        验证给定的着色方案是否有效
        要求：
        1. 所有顶点都已着色
        2. 每条超边包含至少2种不同颜色
        """
        # 检查所有顶点是否都着色了
        if set(coloring.keys()) != set(hg.v):
            return False
        
        # 检查每条超边是否包含至少2种颜色
        for edge in hg.e[0]:
            edge_colors = set(coloring[v] for v in edge)
            if len(edge_colors) < 2:
                return False
        
        return True


class HypergraphHamiltonianPath(BaseTask):
    """任务12：超图哈密顿路径"""
    
    task_name = "HypergraphHamiltonianPath"
    category = "algorithmic_reasoning"
    difficulty = "hard"
    description = "Check if Hamiltonian path exists and validate the path"
    use_guaranteed_solution = True  # 标记：生成时使用保证有解的构造方法
    
    @classmethod
    def prepare_example(
        cls,
        hypergraph: HyperGraph,
        encoding_method: str = 'N-Set',
        vqa_mode: str = 'multimodal',
        guaranteed_path_info: dict = None
    ) -> Dict[str, Any]:
        encoded_text = encode_graph(hypergraph, encoding_method)
        
        # 如果提供了保证的路径信息，直接使用；否则尝试查找
        if guaranteed_path_info is not None:
            source = guaranteed_path_info['source']
            target = guaranteed_path_info['target']
            exists = True
            path = guaranteed_path_info['path']
        elif len(hypergraph.v) < 2:
            source, target = 0, 0
            exists, path = hypergraph.has_hamiltonian_path(source, target)
        else:
            # 尝试找到哈密顿路径
            hamiltonian_found = False
            for _ in range(min(10, len(hypergraph.v))):
                s, t = random.sample(hypergraph.v, 2)
                exists_test, path_test = hypergraph.has_hamiltonian_path(s, t)
                if exists_test:
                    hamiltonian_found = True
                    ham_source, ham_target = s, t
                    ham_path = path_test
                    break
            
            # 50%使用哈密顿路径（Yes），50%随机选择
            if hamiltonian_found and random.random() < 0.5:
                source, target = ham_source, ham_target
                exists, path = True, ham_path
            else:
                source, target = random.sample(hypergraph.v, 2)
                exists, path = hypergraph.has_hamiltonian_path(source, target)
        
        if exists and path:
            answer = f"Yes,Path:[{','.join(f'e{i}' for i in path)}]"
        else:
            answer = "No"
        
        question_text = cls.generate_question_text(
            encoded_text,
            source=source,
            target=target
        )
        
        return {
            'question_text': question_text,
            'answer': f"Ans:{answer}.",
            'ground_truth': {
                'exists': exists,
                'path': path,
                'source': source,
                'target': target,
                'vertices': list(hypergraph.v),
                'hyperedges': [list(edge) for edge in hypergraph.e[0]],
            },
            'hypergraph': hypergraph,
            'source': source,
            'target': target,
        }
    
    @classmethod
    def generate_question_text(cls, encoded_text: str, **kwargs) -> str:
        source = kwargs.get('source', 0)
        target = kwargs.get('target', 1)
        return (
            f"{encoded_text}\n"
            f"Q: Does a Hamiltonian path exist from vertex v{source} to vertex v{target} in hypergraph G? "
            f"(Hamiltonian path = path visiting all vertices exactly once). "
            f"If yes, provide the path. List the answer after 'Ans:' as 'Yes,Path:[e0,e1,...]' or 'No'.\nA: "
        )
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: Any) -> bool:
        """
        验证哈密顿路径预测：
        1. 如果预测No，验证ground_truth确实不存在
        2. 如果预测Yes，提取并验证路径的正确性
        """
        predicted_part = extract_answer_part(predicted)
        
        has_yes = bool(re.search(r'\byes\b', predicted_part))
        has_no = bool(re.search(r'\bno\b', predicted_part))
        
        # 提取ground_truth信息
        if isinstance(ground_truth, dict):
            gt_exists = ground_truth.get('exists', False)
            source = ground_truth.get('source')
            target = ground_truth.get('target')
            vertices = ground_truth.get('vertices', [])
            hyperedges = ground_truth.get('hyperedges', [])
        else:
            gt_exists = 'yes' in str(ground_truth).lower()
            source = target = None
            vertices = hyperedges = []
        
        # 预测No
        if has_no and not has_yes:
            return not gt_exists
        
        # 预测Yes - 必须验证路径
        if has_yes:
            if not gt_exists:
                return False
            
            # 提取路径 - 支持多种格式
            # Path:[e0,e1,...] 或 Path: [e0, e1, ...] 或 path:[0,1,2] 等
            path_match = re.search(r'path\s*[:\-]?\s*[\[{]([^\]\}]+)[\]}]', predicted_part, re.IGNORECASE)
            if not path_match:
                # 尝试更宽松的匹配: path e0 e1 e2
                path_match = re.search(r'path\s*[:\-]?\s*(.+?)(?:\n|$)', predicted_part, re.IGNORECASE)
                if not path_match:
                    return False
            
            path_str = path_match.group(1).strip()
            # 移除括号和其他符号
            path_str = path_str.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '')
            
            try:
                # 提取所有数字 - 支持 e0, e_0, edge0, 0 等格式
                pred_path = []
                # 分割：支持逗号、空格、分号等
                items = re.split(r'[,;\s]+', path_str)
                for item in items:
                    item = item.strip()
                    if not item:
                        continue
                    # 提取数字
                    match = re.search(r'(?:e(?:dge)?[_\-]?)?(\d+)', item, re.IGNORECASE)
                    if match:
                        edge_idx = int(match.group(1))
                        pred_path.append(edge_idx)
                
                if not pred_path:
                    return False
            except:
                return False
            
            if not hyperedges or source is None or target is None:
                return False
            
            return cls._validate_hamiltonian_path_with_gt(vertices, hyperedges, source, target, pred_path)
        
        return False
    
    @classmethod
    def _validate_hamiltonian_path_with_gt(
        cls,
        vertices: list,
        hyperedges: list,
        source: int,
        target: int,
        edge_path: List[int]
    ) -> bool:
        """使用ground_truth信息验证路径的有效性"""
        if not edge_path:
            return False
        
        num_edges = len(hyperedges)
        for edge_idx in edge_path:
            if edge_idx < 0 or edge_idx >= num_edges:
                return False
        
        visited_vertices = set()
        current_vertex = source
        
        for i, edge_idx in enumerate(edge_path):
            edge = hyperedges[edge_idx]
            edge_set = set(edge)
            
            if i == 0:
                if source not in edge_set:
                    return False
                visited_vertices.add(source)
                for v in edge_set:
                    visited_vertices.add(v)
                next_vertices = list(edge_set - {source})
                if next_vertices:
                    current_vertex = next_vertices[0]
            else:
                if current_vertex not in edge_set:
                    return False
                for v in edge_set:
                    visited_vertices.add(v)
                next_vertices = edge_set - {current_vertex}
                if next_vertices:
                    current_vertex = list(next_vertices)[0]
        
        reached_target = (current_vertex == target) or (target in visited_vertices)
        visited_all = (visited_vertices == set(vertices))
        
        return reached_target and visited_all
    
    @classmethod
    def _validate_hamiltonian_path(
        cls, 
        hypergraph: HyperGraph, 
        source: int, 
        target: int, 
        edge_path: List[int]
    ) -> bool:
        """验证路径的有效性（兼容旧代码）"""
        return cls._validate_hamiltonian_path_with_gt(
            list(hypergraph.v),
            [list(edge) for edge in hypergraph.e[0]],
            source,
            target,
            edge_path
        )


# ========== 任务注册表 ==========

TASK_REGISTRY = {
    'VertexCount': VertexCount,
    'HyperedgeCount': HyperedgeCount,
    'SpecificVertexDegreeCount': SpecificVertexDegreeCount,
    'SpecificOrderHyperedgeCount': SpecificOrderHyperedgeCount,
    'DirectNeighbor': DirectNeighbor,
    'NNeighbor': NNeighbor,
    'OrderBasedShortestPath': OrderBasedShortestPath,
    'OrderBasedMaxFlow': OrderBasedMaxFlow,
    'StrictHypercycle': StrictHypercycle,
    'HypergraphIsomorphism': HypergraphIsomorphism,
    'Hypergraph3Coloring': Hypergraph3Coloring,
    'HypergraphHamiltonianPath': HypergraphHamiltonianPath,
}


def get_task(task_name: str) -> BaseTask:
    """获取任务类"""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name]


def get_all_tasks() -> List[str]:
    """获取所有任务名称"""
    return list(TASK_REGISTRY.keys())
