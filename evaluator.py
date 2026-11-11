"""
HyGVL_Bench Evaluation Module

提供VLM评估框架，支持text-only, vision-only, 和 multimodal VQA
支持多种VLM模型：GPT-4o, , Qwen, LLaMA等
"""

import os
import json
import pickle
import time
import base64
from typing import Dict, List, Any, Optional, Callable
from tqdm import tqdm
import numpy as np
import pandas as pd
from openai import OpenAI


class VLMEvaluator:
    """VLM评估器"""
    
    def __init__(
        self, 
        benchmark_dir: str,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化评估器
        
        Args:
            benchmark_dir: 基准测试数据目录
            model_name: 模型名称，如 "gpt-4o", "gpt-3.5-turbo", "Qwen3-VL-8B"等
            api_key: API密钥
            base_url: API基础URL
        """
        self.benchmark_dir = benchmark_dir
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key before evaluating models.")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 延迟加载样本
        self.samples = None
        self.results = []
        
    def _load_samples(self) -> List[Dict]:
        """加载基准测试样本"""
        # 优先加载JSON文件
        json_path = os.path.join(self.benchmark_dir, 'benchmark_samples.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                samples = json.load(f)
            print(f"Loaded {len(samples)} samples from {json_path}")
            return samples
            
        # 备选：加载pickle文件
        pickle_path = os.path.join(self.benchmark_dir, 'benchmark_samples.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"Loaded {len(samples)} samples from {pickle_path}")
            return samples
        
        raise FileNotFoundError(f"No benchmark samples found in {self.benchmark_dir}")
    
    def _load_task_samples(self, task_name: str, vqa_mode: str = 'text_only') -> List[Dict]:
        """按任务加载样本
        
        Args:
            task_name: 任务名称
            vqa_mode: VQA模式 (text_only, vision_only, multimodal)
            
        Returns:
            样本列表
        """
        # 构建路径: data/{vqa_mode}/{task_name}/samples.json
        vqa_mode_map = {
            'text_only': 'Text-Only',
            'vision_only': 'Vision-Only',  
            'multimodal': 'Multimodal'
        }
        
        mode_dir = vqa_mode_map.get(vqa_mode, 'Text-Only')
        samples_path = os.path.join(self.benchmark_dir, mode_dir, task_name, 'samples.json')
        
        if not os.path.exists(samples_path):
            print(f"Warning: {samples_path} not found")
            return []
        
        with open(samples_path, 'r') as f:
            samples = json.load(f)
        
        print(f"Loaded {len(samples)} samples from {samples_path}")
        return samples
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _call_vlm_api(self, question_text: str, image_path: Optional[str] = None) -> str:
        """
        调用VLM API
        
        Args:
            question_text: 问题文本
            image_path: 图像路径（可选）
            
        Returns:
            模型回复
        """
        max_retries = 3
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                if image_path and os.path.exists(image_path):
                    # 多模态输入
                    base64_image = self._encode_image(image_path)
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                else:
                    # 纯文本输入
                    messages = [
                        {"role": "user", "content": question_text}
                    ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"\nAPI调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return "ERROR: API调用失败"
        
        return "ERROR: 超过最大重试次数"
    
    def evaluate(
        self,
        model_fn: Optional[Callable[[str, Optional[str]], str]] = None,
        output_path: Optional[str] = None,
        filter_task: Optional[str] = None,
        filter_category: Optional[str] = None,
        filter_difficulty: Optional[int] = None,
        filter_vqa_mode: Optional[str] = None,
        filter_encoding: Optional[str] = None,
        filter_style: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_builtin_api: bool = True,
        add_vis_instruction: bool = False,
    ) -> Dict[str, Any]:
        """
        评估VLM模型
        
        Args:
            model_fn: 自定义模型推理函数，接受(question_text, image_path)返回答案字符串
                     如果为None且use_builtin_api=True，则使用内置API
            output_path: 结果保存路径
            filter_task: 只评估特定任务
            filter_category: 只评估特定类别
            filter_difficulty: 只评估特定难度
            filter_vqa_mode: 只评估特定VQA模式 (text_only, vision_only, multimodal)
            filter_encoding: 只评估特定编码方式 (N-Set, HO-Inc, etc.)
            filter_style: 只评估特定可视化风格 (spring, kamada_kawai, etc.) - 仅用于vision_only
            max_samples: 最大评估样本数
            use_builtin_api: 是否使用内置API（当model_fn为None时）
            add_vis_instruction: multimodal模式下是否添加visualization instruction（默认False）
            
        Returns:
            评估结果字典
        """
        # 确定使用哪个模型函数
        if model_fn is None:
            if use_builtin_api:
                model_fn = self._call_vlm_api
                print(f"使用内置API调用模型: {self.model_name}")
            else:
                raise ValueError("必须提供model_fn或设置use_builtin_api=True")
        
        # 加载或筛选样本
        if filter_task and filter_vqa_mode:
            # 直接加载特定任务的样本
            eval_samples = self._load_task_samples(filter_task, filter_vqa_mode)
        else:
            # 从已加载的样本中筛选
            if self.samples is None:
                self.samples = self._load_samples()
            eval_samples = self.samples
            
            if filter_task:
                eval_samples = [s for s in eval_samples if s['metadata'].get('task_type', s.get('task_type')) == filter_task]
                print(f"Filtered to task: {filter_task}, {len(eval_samples)} samples")
                
        if filter_category and not (filter_task and filter_vqa_mode):
            eval_samples = [s for s in eval_samples if s['metadata'].get('category', s.get('category')) == filter_category]
            print(f"Filtered to category: {filter_category}, {len(eval_samples)} samples")
            
        if filter_difficulty and not (filter_task and filter_vqa_mode):
            eval_samples = [s for s in eval_samples if s['metadata'].get('difficulty', s.get('difficulty')) == filter_difficulty]
            print(f"Filtered to difficulty: {filter_difficulty}, {len(eval_samples)} samples")
            
        # 只对text-only和multimodal模式过滤encoding（vision-only没有prompt_template）
        if filter_encoding and filter_vqa_mode in ['text_only', 'multimodal']:
            eval_samples = [s for s in eval_samples if s.get('prompt_template', '') == filter_encoding]
            print(f"Filtered to encoding: {filter_encoding}, {len(eval_samples)} samples")
            
        # 对vision-only模式过滤可视化风格
        if filter_style and filter_vqa_mode in ['vision_only', 'multimodal']:
            eval_samples = [s for s in eval_samples if s.get('visualization_config', {}).get('layout', '') == filter_style]
            print(f"Filtered to style: {filter_style}, {len(eval_samples)} samples")
            
        
        if filter_encoding and filter_style and filter_vqa_mode in ['multimodal']:
            eval_samples = [s for s in eval_samples if s.get('prompt_template', '') == filter_encoding and s.get('visualization_config', {}).get('layout', '') == filter_style]
            print(f"Filtered to encoding: {filter_encoding} and style: {filter_style}, {len(eval_samples)} samples")
        
        if max_samples:
            eval_samples = eval_samples[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        print(f"\nEvaluating {len(eval_samples)} samples...")
        
        # 评估每个样本
        results = []
        correct_count = 0
        
        for sample in tqdm(eval_samples, desc="Evaluating"):
            question = sample.get('question', sample.get('question_text', ''))
            text_description = sample.get('text_description', '')
            ground_truth = sample.get('ground_truth', sample.get('answer', ''))
            
            # 确定VQA模式：优先使用filter参数，其次从metadata获取
            if filter_vqa_mode:
                vqa_mode = filter_vqa_mode
            elif 'metadata' in sample and 'vqa_mode' in sample['metadata']:
                vqa_mode = sample['metadata']['vqa_mode']
            else:
                vqa_mode = sample.get('vqa_mode', 'text_only')  # 默认text_only
            
            # 根据VQA模式选择输入
            if vqa_mode == 'vision_only':
                # 仅视觉：提供图像+问题文本
                # image_path已经是相对于工作目录的完整路径（如data/Vision-Only/...）
                image_path = sample.get('image_path', '')
                predicted_answer = model_fn(question, image_path)
            elif vqa_mode == 'multimodal':
                # 多模态：图像+文本（text_description + question）
                image_path = sample.get('image_path', '')
                
                # 根据add_vis_instruction参数决定是否添加visualization instruction
                if add_vis_instruction:
                    # 保持原有question（包含visualization instruction）
                    full_text = text_description + "\n" + question if text_description else question
                else:
                    # 移除visualization instruction，只保留实际问题
                    # question格式: "instruction...\n\nQ: actual_question"
                    if "\n\nQ: " in question or "\n\n" in question:
                        # 提取实际问题部分
                        parts = question.split("\n\n")
                        actual_question = parts[-1]  # 取最后一部分
                        if actual_question.startswith("Q: "):
                            actual_question = actual_question[3:]  # 移除"Q: "前缀
                        full_text = text_description + "\n" + actual_question if text_description else actual_question
                    else:
                        # 没有instruction，直接使用
                        full_text = text_description + "\n" + question if text_description else question
                
                predicted_answer = model_fn(full_text, image_path)
            else:  # text_only
                # 仅文本：text_description + question
                full_text = text_description + "\n" + question if text_description else question
                predicted_answer = model_fn(full_text, None)
            
            # 检查答案
            task_type = sample.get('task_type', sample.get('metadata', {}).get('task_type', ''))
            is_correct = self._check_answer(predicted_answer, ground_truth, task_type)
            
            if is_correct:
                correct_count += 1
            
            # 从hypergraph数据中提取nvertices和nedges
            nvertices = 0
            nedges = 0
            
            # 优先从hypergraph字段获取
            if 'hypergraph' in sample:
                hg_data = sample['hypergraph']
                nvertices = len(hg_data.get('vertices', []))
                nedges = len(hg_data.get('hyperedges', []))
            else:
                # 备用：从text_description提取
                import re
                text_desc = sample.get('text_description', '')
                vertex_match = re.search(r'vertices (v\d+(?:,\s*v\d+)*(?:,?\s*and\s+v\d+)?)', text_desc)
                if vertex_match:
                    vertices_str = vertex_match.group(1)
                    vertex_ids = re.findall(r'v(\d+)', vertices_str)
                    nvertices = len(vertex_ids)
                edge_match = re.search(r'hyperedges (e\d+(?:,\s*e\d+)*(?:,?\s*and\s+e\d+)?)', text_desc)
                if edge_match:
                    edges_str = edge_match.group(1)
                    edge_ids = re.findall(r'e(\d+)', edges_str)
                    nedges = len(edge_ids)
            
            # 获取text_encoding和visual_style
            # multimodal: 两者都有
            # text_only: 只有text_encoding，visual_style留空
            # vision_only: 只有visual_style，text_encoding留空
            if vqa_mode == 'multimodal':
                text_encoding = sample.get('prompt_template', '')
                visual_style = sample.get('visualization_config', {}).get('layout', '')
            elif vqa_mode == 'text_only':
                text_encoding = sample.get('prompt_template', '')
                visual_style = ''
            else:  # vision_only
                text_encoding = ''
                visual_style = sample.get('visualization_config', {}).get('layout', '')
            
            # 构建metadata
            metadata = {
                'task_type': sample.get('task_type', ''),
                'category': sample.get('category', sample.get('metadata', {}).get('category', '')),
                'difficulty': sample.get('difficulty', sample.get('metadata', {}).get('difficulty', 0)),
                'vqa_mode': vqa_mode,
                'text_encoding': text_encoding,
                'visual_style': visual_style,
                'nvertices': nvertices,
                'nedges': nedges
            }
            
            results.append({
                'sample_id': sample.get('id', sample.get('sample_id', len(results))),
                'question': question,
                'text_description': sample.get('text_description', ''),
                'full_input': full_text,  # 使用实际发送给模型的文本
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'metadata': metadata
            })
        
        # 计算统计
        accuracy = correct_count / len(eval_samples) if eval_samples else 0
        
        stats = {
            'total_samples': len(eval_samples),
            'correct_count': correct_count,
            'overall_accuracy': accuracy,
            'samples': results
        }
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        self.results = results
        return stats
    
    def _check_answer(self, predicted: str, ground_truth: Any, task_type: str) -> bool:
        """
        检查答案是否正确
        
        Args:
            predicted: 预测答案
            ground_truth: 正确答案
            task_type: 任务类型
            
        Returns:
            是否正确
        """
        # 尝试使用任务类的check_answer方法
        try:
            from tasks import get_task
            task_class = get_task(task_type)
            return task_class.check_answer(predicted, ground_truth)
        except:
            pass
        
        # 备用方案：通用检查逻辑
        pred_clean = str(predicted).strip().lower().replace('.', '').replace(',', '').replace(':', ' ')
        truth_str = str(ground_truth).strip().lower().replace('.', '').replace(',', '').replace(':', ' ')
        
        # 数值类任务
        if task_type in ['VertexCount', 'HyperedgeCount', 'SpecificVertexDegreeCount', 
                        'SpecificOrderHyperedgeCount', 
                        # 'HypergraphVertexCover', 
                        'OrderBasedMaxFlow']:
            import re
            pred_nums = re.findall(r'\d+', pred_clean)
            if pred_nums:
                try:
                    return int(pred_nums[-1]) == int(ground_truth)
                except:
                    pass
        
        # 集合类任务
        if task_type in ['DirectNeighbor', 'NNeighbor']:
            import re
            if "no neighbor" in pred_clean:
                return isinstance(ground_truth, (list, set)) and len(ground_truth) == 0
            pred_nums = set(re.findall(r'\d+', pred_clean))
            if isinstance(ground_truth, (list, set)):
                truth_nums = set(str(v) for v in ground_truth)
                return pred_nums == truth_nums
        
        # 路径/结构类任务
        if task_type in ['OrderBasedShortestPath', 'StrictHypercycle', 'HypergraphHamiltonianPath']:
            import re
            # 检查权重/顶点数
            weight_match = re.search(r'(weight|vertices)[:\s]*(\d+)', pred_clean)
            if weight_match and isinstance(ground_truth, dict):
                key = 'weight' if 'weight' in weight_match.group(1) else 'vertices'
                try:
                    return int(weight_match.group(2)) == ground_truth.get(key, -1)
                except:
                    pass
            
            # 对于哈密顿路径，需要验证路径的有效性
            if task_type == 'HypergraphHamiltonianPath' and isinstance(ground_truth, dict):
                # 提取模型给出的路径
                path_match = re.search(r'path[:\s]*\[([^\]]+)\]', pred_clean)
                
                if "no" in pred_clean and not path_match:
                    # 模型说不存在，检查ground truth
                    return not ground_truth.get('exists', True)
                elif ("yes" in pred_clean or path_match) and ground_truth.get('exists', True):
                    # 模型说存在，需要验证路径
                    if not path_match:
                        # 说Yes但没给路径，认为错误
                        return False
                    
                    # 提取路径中的边
                    path_str = path_match.group(1)
                    # 匹配 e0, e1, e2 这样的边标识
                    pred_edges = re.findall(r'e(\d+)', path_str)
                    if not pred_edges:
                        return False
                    
                    pred_edges = [int(e) for e in pred_edges]
                    
                    # 验证路径是否有效
                    vertices = set(ground_truth.get('vertices', []))
                    hyperedges = ground_truth.get('hyperedges', [])
                    source = ground_truth.get('source')
                    target = ground_truth.get('target')
                    
                    if not vertices or not hyperedges:
                        return False
                    
                    # 使用DFS验证路径是否能访问所有顶点
                    def verify_path(edges, start, end):
                        """验证给定的边序列是否能形成有效的哈密顿路径"""
                        visited = {start}
                        
                        def dfs(current, edge_idx):
                            if edge_idx >= len(edges):
                                # 所有边都用完了，检查是否到达目标并访问所有顶点
                                return current == end and len(visited) == len(vertices)
                            
                            e_idx = edges[edge_idx]
                            if e_idx >= len(hyperedges):
                                return False
                            
                            edge = set(hyperedges[e_idx])
                            
                            # 当前顶点必须在这条边中
                            if current not in edge:
                                return False
                            
                            # 尝试移动到边中的每个顶点
                            for next_v in edge:
                                if next_v not in visited:
                                    visited.add(next_v)
                                    if dfs(next_v, edge_idx + 1):
                                        return True
                                    visited.remove(next_v)
                                elif edge_idx + 1 < len(edges):
                                    # 如果已访问但不是最后一条边，可以在边内移动
                                    if dfs(next_v, edge_idx + 1):
                                        return True
                            
                            return False
                        
                        return dfs(start, 0)
                    
                    return verify_path(pred_edges, source, target)
                elif "yes" in pred_clean and not ground_truth.get('exists', True):
                    # 模型说存在但实际不存在
                    return False
                else:
                    return False
            
            # 其他路径任务的检查
            if "no" in pred_clean:
                if isinstance(ground_truth, dict):
                    return not ground_truth.get('exists', True)
            elif "yes" in pred_clean:
                if isinstance(ground_truth, dict):
                    return ground_truth.get('exists', True)
        
        # 图着色任务
        if task_type == 'Hypergraph3Coloring':
            import re
            # 检查是否可着色
            if "yes" in pred_clean or "can be colored" in pred_clean or "is 3-colorable" in pred_clean:
                if isinstance(ground_truth, dict):
                    return ground_truth.get('is_colorable', False)
                return str(ground_truth).lower() in ['yes', 'true', '1']
            elif "no" in pred_clean or "cannot be colored" in pred_clean or "not 3-colorable" in pred_clean:
                if isinstance(ground_truth, dict):
                    return not ground_truth.get('is_colorable', True)
                return str(ground_truth).lower() in ['no', 'false', '0']
        
        # 同构任务
        if task_type == 'HypergraphIsomorphism':
            import re
            if "yes" in pred_clean or "isomorphic" in pred_clean:
                if isinstance(ground_truth, dict):
                    return ground_truth.get('is_isomorphic', False)
                return str(ground_truth).lower() in ['yes', 'true', '1']
            elif "no" in pred_clean or "not isomorphic" in pred_clean:
                if isinstance(ground_truth, dict):
                    return not ground_truth.get('is_isomorphic', True)
                return str(ground_truth).lower() in ['no', 'false', '0']
        
        # 默认：字符串匹配
        return pred_clean == truth_str
    
    def print_statistics(self):
        """打印评估统计"""
        if not self.results:
            print("No evaluation results yet!")
            return
        
        # 按任务统计
        task_stats = {}
        for result in self.results:
            task = result['metadata']['task_type']
            if task not in task_stats:
                task_stats[task] = {'correct': 0, 'total': 0}
            task_stats[task]['total'] += 1
            if result['is_correct']:
                task_stats[task]['correct'] += 1
        
        print("\n" + "="*60)
        print("Evaluation Statistics")
        print("="*60)
        
        print("\nPer-task Accuracy:")
        for task, stats in sorted(task_stats.items()):
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {task:35s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
        
        # 总体统计
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        overall_acc = correct / total * 100
        
        print("\n" + "-"*60)
        print(f"Overall Accuracy: {overall_acc:.1f}% ({correct}/{total})")
        print("="*60 + "\n")


def evaluate_with_gpt4o(
    benchmark_dir: str,
    output_path: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用GPT-4o评估的便捷函数
    
    Args:
        benchmark_dir: 基准测试数据目录
        output_path: 结果保存路径
        api_key: API密钥（可选）
        base_url: API基础URL（可选）
        
    Returns:
        评估统计
    """
    evaluator = VLMEvaluator(
        benchmark_dir=benchmark_dir,
        model_name="gpt-4o",
        api_key=api_key,
        base_url=base_url
    )
    
    stats = evaluator.evaluate(
        use_builtin_api=True,
        output_path=output_path
    )
    
    evaluator.print_statistics()
    return stats


def evaluate_with_custom_model(
    benchmark_dir: str,
    model_name: str,
    output_path: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用自定义模型评估的便捷函数
    
    Args:
        benchmark_dir: 基准测试数据目录
        model_name: 模型名称
        output_path: 结果保存路径
        api_key: API密钥（可选）
        base_url: API基础URL（可选）
        
    Returns:
        评估统计
    """
    evaluator = VLMEvaluator(
        benchmark_dir=benchmark_dir,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url
    )
    
    stats = evaluator.evaluate(
        use_builtin_api=True,
        output_path=output_path
    )
    
    evaluator.print_statistics()
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VLMBench Evaluator - 评估VLM性能并生成CSV')
    parser.add_argument('--benchmark_dir', type=str, default='data',
                        help='基准测试数据目录（默认：data）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录（默认自动生成：results/{model}/{vqa_mode}）')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='模型名称（默认：gpt-4o）')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API密钥')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API基础URL')
    parser.add_argument('--vqa_mode', type=str, default='multimodal',
                        choices=['text_only', 'vision_only', 'multimodal'],
                        help='VQA模式（默认：multimodal）')
    parser.add_argument('--encoding', type=str, default=None,
                        help='文本编码方法（不指定则评估所有编码）')
    parser.add_argument('--style', type=str, default=None,
                        help='可视化风格（不指定则评估所有风格）')
    parser.add_argument('--task', type=str, default=None,
                        help='评估特定任务（不指定则评估所有任务）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大评估样本数（每个任务）')
    parser.add_argument('--with_vis_instruction', action='store_true',
                        help='multimodal模式下添加visualization instruction（默认不添加）')
    
    args = parser.parse_args()
    
    # 自动生成output_dir
    if args.output_dir is None:
        model_name_clean = args.model.replace('/', '_').replace(':', '_')
        args.output_dir = f'results/{model_name_clean}/{args.vqa_mode}'
        print(f"使用自动生成的输出目录: {args.output_dir}")
    
    # 任务名称映射（使用任务本身的名称作为CSV文件名）
    TASK_NAME_MAPPING = {
        'DirectNeighbor': 'DirectNeighbor',
        'NNeighbor': 'NNeighbor',
        'OrderBasedShortestPath': 'OrderBasedShortestPath',
        'VertexCount': 'VertexCount',
        'HyperedgeCount': 'HyperedgeCount',
        'SpecificVertexDegreeCount': 'SpecificVertexDegreeCount',
        'SpecificOrderHyperedgeCount': 'SpecificOrderHyperedgeCount',
        'HypergraphIsomorphism': 'HypergraphIsomorphism',
        'StrictHypercycle': 'StrictHypercycle',
        'OrderBasedMaxFlow': 'OrderBasedMaxFlow',
        'Hypergraph3Coloring': 'Hypergraph3Coloring',
        'HypergraphHamiltonianPath': 'HypergraphHamiltonianPath',
    }
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = VLMEvaluator(
        benchmark_dir=args.benchmark_dir,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 确定要评估的任务列表
    if args.task:
        tasks_to_eval = [args.task]
    else:
        # 评估所有任务
        tasks_to_eval = list(TASK_NAME_MAPPING.keys())
    
    # 评估每个任务并生成CSV
    all_results = {}
    
    for task_name in tasks_to_eval:
        print(f"\n{'='*60}")
        print(f"评估任务: {task_name} ({args.vqa_mode})")
        print(f"{'='*60}")
        
        try:
            # 运行评估
            # 对于multimodal模式，如果没有指定encoding和style，则评估所有样本
            # 否则按指定的encoding/style筛选
            stats = evaluator.evaluate(
                use_builtin_api=True,
                filter_task=task_name,
                filter_vqa_mode=args.vqa_mode,
                filter_encoding=args.encoding if args.encoding != 'N-Set' else None,  # 如果是默认N-Set则不筛选
                filter_style=args.style,
                max_samples=args.max_samples,
                add_vis_instruction=args.with_vis_instruction,  # 传递参数，默认False
            )
            
            all_results[task_name] = stats
            
            # 生成CSV
            csv_data = []
            for idx, sample in enumerate(stats['samples']):
                row = {
                    '': idx,
                    'id': sample['sample_id'],
                    'question': sample.get('full_input', ''),
                    'answer': sample['ground_truth'],
                    'algorithm': '',
                    'text_encoding': sample['metadata'].get('text_encoding', ''),
                    'visual_style': sample['metadata'].get('visual_style', ''),
                    'nvertices': sample['metadata'].get('nvertices', 0),
                    'nedges': sample['metadata'].get('nedges', 0),
                    'response': sample['predicted_answer'],
                    'result': sample['is_correct']
                }
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            # CSV文件名：只包含任务名，不包含encoding或style
            csv_filename = f"{TASK_NAME_MAPPING.get(task_name, task_name.lower())}.csv"
            csv_path = os.path.join(args.output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            
            print(f"\n任务 {task_name} 统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  正确数: {stats['correct_count']}")
            print(f"  准确率: {stats['overall_accuracy']:.2%}")
            print(f"  CSV保存至: {csv_path}")
        except Exception as e:
            print(f"\n错误: 评估任务 {task_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印总体统计
    print(f"\n{'='*60}")
    print(f"总体评估结果 ({args.vqa_mode})")
    print(f"{'='*60}")
    
    overall_stats = []
    for task_name, results in all_results.items():
        overall_stats.append({
            'Task': task_name,
            'Total': results['total_samples'],
            'Correct': results['correct_count'],
            'Accuracy': f"{results['overall_accuracy']:.2%}"
        })
    
    stats_df = pd.DataFrame(overall_stats)
    print(stats_df.to_string(index=False))
    
    # 保存总体统计
    stats_path = os.path.join(args.output_dir, f'overall_stats_{args.vqa_mode}.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n总体统计已保存至: {stats_path}")

'''
python evaluator.py \
  --benchmark_dir data \
  --output_dir results/gpt4o/text_only \
  --model gpt-4o \
  --vqa_mode text_only \
  --encoding N-Set \
  --max_samples 5

python evaluator.py \
  --benchmark_dir data \
  --output_dir results/gpt4o/vision_only \
  --model gpt-4o \
  --vqa_mode vision_only \
  --max_samples 15
  
python evaluator.py \
  --benchmark_dir data \
  --output_dir results/gpt4o/multimodal \
  --model gpt-4o \
  --vqa_mode multimodal \
  --encoding N-Set \
  --style shell
  --max_samples 5
'''
