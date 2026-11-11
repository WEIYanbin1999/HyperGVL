# HyperGVL

[EMNLP 2026] HyperGVL: Benchmarking and Improving Large Vision-Language Models in Hypergraph Understanding and Reasoning

The first comprehensive benchmark for evaluating Vision-Language Models (VLMs) on hypergraph understanding and reasoning tasks.

## 📋 Task Overview

VLMBench contains **12 tasks** divided into **2 categories** with **3 difficulty levels**.

### Task Categories

#### 🔍 **Structure Understanding** (6 tasks)
Tasks that require reading and interpreting basic hypergraph structure.

| Task | Difficulty | Description |
|------|-----------|-------------|
| **VertexCount** | Easy | Count total number of vertices |
| **HyperedgeCount** | Easy | Count total number of hyperedges |
| **DirectNeighbor** | Easy | Identify direct neighbors of a vertex |
| **SpecificVertexDegreeCount** | Medium | Count vertices with specific degree |
| **SpecificOrderHyperedgeCount** | Medium | Count hyperedges with specific order |
| **NNeighbor** | Medium | Find neighbors via hyperedges with order ≥ n |

#### 🧠 **Algorithmic Reasoning** (6 tasks)
Tasks that require complex graph algorithms and multi-step reasoning.

| Task | Difficulty | Description |
|------|-----------|-------------|
| **OrderBasedShortestPath** | Medium | Shortest path with hyperedge order as weight |
| **OrderBasedMaxFlow** | Hard| Maximum flow with hyperedge order as capacity |
| **StrictHypercycle** | Medium | Find a strict hypercycle (adjacent edges share exactly 1 vertex) |
| **HypergraphIsomorphism** | Hard | Determine if two hypergraphs are isomorphic |
| **Hypergraph3Coloring** | Hard | 3-color hypergraph (each edge has ≥2 colors, NP-complete) |
| **HypergraphHamiltonianPath** | Hard | Check Hamiltonian path existence (NP-complete) |

### Difficulty Levels

- **Easy** (3 tasks): Direct observation, single-step operations
- **Medium** (5 tasks): Filtered computation, 2-3 step reasoning
- **Hard** (4 tasks): Complex algorithms, NP-complete problems, deep reasoning

### Hypergraph Language
7 Types of Hypergraph Languages

### Hypergraph Visualization
5 Types of Hypergraph Visualization







## 🎯 Benchmark Modes

VLMBench supports
### Evaluating both LLMs or VLMs
LLM: Text-Only Mode
VLM: Multimodal Mode

## 📊 Dataset Statistics

With the default 200 native samples per task and all 12 tasks, the generator
produces the following dataset sizes when every stage completes:

| Mode | Variants per native sample | Total samples |
|------|----------------------------|---------------|
| Native | 1 | 2,400 |
| Text-Only | 7 text encodings | 16,800 |
| Vision-Only | 5 visual layouts | 12,000 |
| Multimodal | 7 × 5 combinations | 84,000 |

Variants share the same underlying graph, question, and answer. Multimodal
samples reuse the images created by the Vision-Only stage. Native samples use
a small:medium:large ratio of 1:2:1, with 5–10, 10–15, and 15–20 vertices,
respectively. Generated datasets are not included in this repository.

## Setup

Run the scripts from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

SciPy is required at generator startup, including when using `--mode native`,
because the generator imports the visualization module.

## Generate a benchmark

Start with a small example:

```bash
python generate_benchmark.py --mode all \
  --tasks VertexCount HyperedgeCount \
  --num_samples 4 --output_base ./data_smoke --seed 42
```

Generate the default full benchmark:

```bash
python generate_benchmark.py --mode all --num_samples 200 --output_base ./data
```

The available modes are `native`, `text-only`, `vision-only`, `multimodal`, and
`all`. Individual text and vision stages require existing Native data; the
multimodal stage also requires existing Vision-Only data. Each mode writes
`<output_base>/<Mode>/<Task>/samples.json`, and Vision-Only also writes PNG
images. The main generator periodically saves progress and skips existing
sample IDs when resuming. Use a fresh output directory when changing the
generation configuration.

`generate_binary_tasks.py` is a separate script that writes 200 balanced
isomorphism examples to `data/Native/HypergraphIsomorphism/samples.json`,
overwriting that file. It currently implements only isomorphism generation.

## Evaluate a model

Configure `OPENAI_API_KEY` in the environment before evaluation. Set
`OPENAI_BASE_URL` when using a compatible model endpoint. Dataset generation
does not require an API key or a model endpoint.

```bash
python evaluator.py --benchmark_dir data \
  --model gpt-4o --vqa_mode text_only \
  --task VertexCount --max_samples 5
```

The evaluator writes per-task CSV files and an overall accuracy CSV under
`results/<model>/<mode>/` unless `--output_dir` is supplied.

## Current limitations

- Isomorphism negative examples are randomly reconstructed and labeled `No`
  without an isomorphism check; some labels can be incorrect.
- The guaranteed cycle and Hamiltonian path constructors can produce reference
  answers rejected by the task validators. These constructors and validators
  need further validation before using their outputs as trusted ground truth.
- The main generator constructs positive examples for cycle, coloring, and
  Hamiltonian path tasks; it does not balance their positive and negative labels.
- `N-Pair`, `LO-Inc`, and `Adj-Mat` describe pairwise adjacency and can map
  different hypergraphs to the same text, losing higher-order structure.
- The separate binary generator stores the second graph inside `metadata`,
  while the main conversion stages expect a top-level `hypergraph_H` field.
- `evaluator_pl.py` expects a different aggregate dataset schema. The package
  exports in `__init__.py` also refer to older modules; use the script entry
  points shown above.

`python test_parser_robustness.py` runs format-parsing examples. It does not
validate dataset labels, graph constructors, or the complete evaluation flow.
