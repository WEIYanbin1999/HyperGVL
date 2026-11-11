"""
HyGVL_Bench Evaluation Module (Parallel Version)

支持 text-only / vision-only / multimodal
支持 GPT-4o / Qwen / LLaMA / 自定义 API
"""

import os
import json
import pickle
import time
import base64
import math
from typing import Dict, List, Any, Optional, Callable
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI


# =========================================================
# 并行 worker（必须是顶层函数）
# =========================================================

def _eval_batch_worker(
    batch_samples: List[Dict],
    model_name: str,
    api_key: str,
    base_url: str,
    add_vis_instruction: bool,
):
    client = OpenAI(api_key=api_key, base_url=base_url)

    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def call_api(text, image_path=None):
        if image_path and os.path.exists(image_path):
            b64 = encode_image(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }]
        else:
            messages = [{"role": "user", "content": text}]

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return resp.choices[0].message.content

    outputs = []

    for sample in batch_samples:
        question = sample.get('question', sample.get('question_text', ''))
        text_desc = sample.get('text_description', '')
        vqa_mode = sample.get('metadata', {}).get('vqa_mode', sample.get('vqa_mode', 'text_only'))

        if vqa_mode == 'vision_only':
            full_text = question
            image_path = sample.get('image_path')
        elif vqa_mode == 'multimodal':
            image_path = sample.get('image_path')
            if add_vis_instruction:
                full_text = text_desc + "\n" + question if text_desc else question
            else:
                q = question.split("\n\n")[-1]
                if q.startswith("Q: "):
                    q = q[3:]
                full_text = text_desc + "\n" + q if text_desc else q
        else:
            image_path = None
            full_text = text_desc + "\n" + question if text_desc else question

        try:
            pred = call_api(full_text, image_path)
        except Exception as e:
            pred = f"ERROR: {e}"

        outputs.append({
            "sample": sample,
            "full_input": full_text,
            "predicted_answer": pred
        })

    return outputs


# =========================================================
# Evaluator
# =========================================================

class VLMEvaluator:

    def __init__(self, benchmark_dir, model_name, api_key=None, base_url=None):
        self.benchmark_dir = benchmark_dir
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.samples = None
        self.results = []

    def _load_samples(self):
        json_path = os.path.join(self.benchmark_dir, "benchmark_samples.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)
        raise FileNotFoundError("benchmark_samples.json not found")

    def evaluate(
        self,
        filter_task=None,
        filter_vqa_mode=None,
        max_samples=None,
        add_vis_instruction=False,
        num_workers=6,
    ):
        if self.samples is None:
            self.samples = self._load_samples()

        eval_samples = self.samples
        if filter_task:
            eval_samples = [
                s for s in eval_samples
                if s['metadata'].get('task_type') == filter_task
            ]

        if max_samples:
            eval_samples = eval_samples[:max_samples]

        print(f"\nEvaluating {len(eval_samples)} samples with {num_workers} workers")

        batch_size = math.ceil(len(eval_samples) / num_workers)
        batches = [
            eval_samples[i:i + batch_size]
            for i in range(0, len(eval_samples), batch_size)
        ]

        all_outputs = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _eval_batch_worker,
                    batch,
                    self.model_name,
                    self.api_key,
                    self.base_url,
                    add_vis_instruction
                )
                for batch in batches
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
                all_outputs.extend(f.result())

        results = []
        correct = 0

        for item in all_outputs:
            sample = item["sample"]
            pred = item["predicted_answer"]
            gt = sample.get("ground_truth", sample.get("answer"))
            task = sample.get("metadata", {}).get("task_type")

            is_correct = str(pred).strip().lower() == str(gt).strip().lower()
            correct += int(is_correct)

            results.append({
                "sample_id": sample.get("id"),
                "full_input": item["full_input"],
                "ground_truth": gt,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "metadata": sample.get("metadata", {})
            })

        self.results = results

        return {
            "total_samples": len(results),
            "correct_count": correct,
            "overall_accuracy": correct / len(results),
            "samples": results
        }


# =========================================================
# CLI 入口（只用 __main__）
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("HyGVL_Bench Parallel Evaluator")
    parser.add_argument("--benchmark_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--vqa_mode", default="multimodal")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--with_vis_instruction", action="store_true")

    args = parser.parse_args()

    if args.output_dir is None:
        model_clean = args.model.replace("/", "_")
        args.output_dir = f"results/{model_clean}/{args.vqa_mode}"

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = VLMEvaluator(
        benchmark_dir=args.benchmark_dir,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )

    tasks = [args.task] if args.task else sorted({
        s["metadata"]["task_type"] for s in evaluator._load_samples()
    })

    summary = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating task: {task}")
        print(f"{'='*60}")

        stats = evaluator.evaluate(
            filter_task=task,
            filter_vqa_mode=args.vqa_mode,
            max_samples=args.max_samples,
            add_vis_instruction=args.with_vis_instruction,
            num_workers=args.num_workers,
        )

        df = pd.DataFrame([
            {
                "id": s["sample_id"],
                "question": s["full_input"],
                "answer": s["ground_truth"],
                "response": s["predicted_answer"],
                "result": s["is_correct"],
            }
            for s in stats["samples"]
        ])

        csv_path = os.path.join(args.output_dir, f"{task}.csv")
        df.to_csv(csv_path, index=False)

        summary.append({
            "Task": task,
            "Total": stats["total_samples"],
            "Correct": stats["correct_count"],
            "Accuracy": f"{stats['overall_accuracy']:.2%}"
        })

        print(f"Saved: {csv_path}")

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, "overall_stats.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nOverall Results:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved overall stats to {summary_path}")
