#!/usr/bin/env python
"""
为所有任务添加get_question_only方法和question_only字段
"""

import re

# 读取tasks.py
with open('tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义每个任务的纯问题文本（从generate_question_text中提取）
task_questions = {
    'VertexCount': 'How many vertices are in the hypergraph G? List the answer after \'Ans:\'.',
    'HyperedgeCount': 'How many hyperedges are in the hypergraph G? List the answer after \'Ans:\'.',
    'SpecificVertexDegreeCount': 'How many vertices have degree {target_degree} in hypergraph G? List the answer after \'Ans:\'.',
    'SpecificOrderHyperedgeCount': 'How many hyperedges have order {target_order} in hypergraph G? List the answer after \'Ans:\'.',
    'DirectNeighbor': 'What are the direct neighbors of vertex v{target_vertex} in hypergraph G? List the answer after \'Ans:\'.',
    'NNeighbor': 'What are the neighbors of vertex v{source_vertex} through hyperedges with order >= {min_order}? List the answer after \'Ans:\'.',
    'OrderBasedShortestPath': 'What is the shortest path length from vertex v{source_vertex} to vertex v{target_vertex} in hypergraph G, where each hyperedge\'s weight equals its order (number of vertices)? List the answer after \'Ans:\'.',
    'OrderBasedMaxFlow': 'What is the maximum flow from vertex v{source_vertex} to vertex v{target_vertex} in hypergraph G, using hyperedge order as capacity? List the answer after \'Ans:\'.',
    'StrictHypercycle': 'Does hypergraph G contain a hypercycle? (A hypercycle is a closed path using different hyperedges, where adjacent hyperedges share exactly one vertex). List the answer after \'Ans:\' in the format [Yes/No].',
    'HypergraphIsomorphism': 'Are these two hypergraphs isomorphic? (Two hypergraphs are isomorphic if there exists a vertex relabeling that transforms one into the other). List the answer after \'Ans:\' in the format [Yes/No].',
    'HypergraphVertexCover': 'What is the minimum vertex cover size in hypergraph G? List the answer after \'Ans:\'.',
    'HypergraphHamiltonianPath': 'Does hypergraph G have a Hamiltonian path from vertex v{source_vertex} to vertex v{target_vertex}? List the answer (Yes/No) after \'Ans:\'.',
}

# 为每个任务添加get_question_only方法
for task_name, question in task_questions.items():
    # 查找该任务的generate_question_text方法
    pattern = rf'(class {task_name}\(BaseTask\):.*?@classmethod\s+def generate_question_text.*?\n(?:.*?\n)*?.*?return \(.*?\n(?:.*?\n)*?.*?\))\s+(@classmethod\s+def check_answer)'
    
    def add_get_question_only(match):
        before = match.group(1)
        after = match.group(2)
        
        # 检查是否需要参数
        if '{' in question:
            # 提取参数名
            params = re.findall(r'\{(\w+)\}', question)
            param_str = ', '.join([f'{p}=None' for p in params])
            param_doc = '\n'.join([f'            {p}: 参数值' for p in params])
            
            get_question_only = f'''
    
    @classmethod
    def get_question_only(cls, **kwargs) -> str:
        """获取纯问题文本（不包含图描述）"""
        return "{question}".format(**kwargs)
    '''
        else:
            get_question_only = f'''
    
    @classmethod
    def get_question_only(cls, **kwargs) -> str:
        """获取纯问题文本（不包含图描述）"""
        return "{question}"
    '''
        
        return before + get_question_only + '\n    ' + after
    
    content = re.sub(pattern, add_get_question_only, content, flags=re.DOTALL)

# 保存修改后的内容
with open('tasks_updated.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("已创建tasks_updated.py")
print("请检查后替换原文件")
