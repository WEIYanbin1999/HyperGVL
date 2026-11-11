"""
改进的可视化说明 - 基于实验验证

删除了无效的彩色区域可视化，采用VLM可理解的表示方法
"""

# ========== 有效的可视化方法及其说明 ==========

VISUALIZATION_INSTRUCTIONS = {
    # 验证有效的方法
    'shell': """In this shell layout bipartite hypergraph visualization:
- VERTICES are shown as BLACK CIRCLES with white labels (v0, v1, v2, etc.) arranged in the INNER ring
- HYPEREDGES are represented as COLORED SQUARE NODES with white labels (e0, e1, e2, etc.) in the OUTER ring
- Each SQUARE NODE represents ONE hyperedge
- COLORED LINES connect each hyperedge square to ALL vertices it contains
- All lines of the SAME COLOR belong to the same hyperedge
- The shell layout arranges nodes in concentric circles for clear visualization
- To count hyperedges: count the number of SQUARE NODES in the outer ring
- To find members of a hyperedge: trace all colored lines from that square to connected circles

Example: If you see a green square 'e2' in the outer ring with lines connecting to v1, v3, v5 in the inner ring, then hyperedge e2 = {v1, v3, v5}
This is a bipartite representation where hyperedges are explicit nodes, similar to bipartite_horizontal and bipartite_radial.""",
    
    'bipartite_horizontal': """In this bipartite hypergraph visualization:
- VERTICES are shown as BLACK CIRCLES with white labels (v0, v1, v2, etc.) in the TOP row
- HYPEREDGES are represented as COLORED SQUARE NODES with white labels (e0, e1, e2, etc.) in the BOTTOM row
- Each SQUARE NODE represents ONE hyperedge
- COLORED LINES connect each hyperedge square to ALL vertices it contains
- All lines of the SAME COLOR belong to the same hyperedge
- To find members of hyperedge e0: look at the square labeled 'e0' and follow all connected lines to circles

Example: If you see a red square 'e0' connected to circles 'v1', 'v3', 'v5', then hyperedge e0 = {v1, v3, v5}""",
    
    'bipartite_radial': """In this bipartite hypergraph visualization:
- VERTICES are shown as BLACK CIRCLES with white labels (v0, v1, etc.) arranged in the OUTER ring
- HYPEREDGES are represented as COLORED SQUARE NODES with white labels (e0, e1, etc.) in the INNER ring
- Each SQUARE represents ONE hyperedge
- COLORED LINES radiate from each hyperedge square to its member vertices
- All lines of the SAME COLOR indicate the same hyperedge
- To count hyperedges: count the number of SQUARE NODES in the inner ring
- To find members of a hyperedge: trace all colored lines from that square to connected circles

Example: If square 'e2' (green) has lines connecting to v0, v2, v4, v7, then hyperedge e2 = {v0, v2, v4, v7}""",
    
    'labeled_regions': """In this hypergraph visualization:
- VERTICES are shown as black circles with white labels (v0, v1, v2, etc.)
- HYPEREDGES are represented by colored polygonal regions
- Each colored region has a WHITE LABEL BOX in its center showing the hyperedge ID (e0, e1, e2, etc.)
- The label 'eX' explicitly identifies that colored region as hyperedge X
- All vertices enclosed by a colored region belong to that hyperedge
- To count hyperedges: count the number of white label boxes showing 'e0', 'e1', 'e2', etc.
- For 2-vertex hyperedges: look for colored lines with a label box in the middle

Example: If you see a blue region with label 'e3' enclosing vertices v2, v5, v8, then hyperedge e3 = {v2, v5, v8}""",
    
    'star_expansion': """In this star expansion visualization:
- VERTICES are shown as BLACK CIRCLES with white labels (v0, v1, etc.) arranged in the OUTER ring
- HYPEREDGES are represented as COLORED SQUARE NODES with white labels (e0, e1, etc.) in the INNER ring  
- COLORED LINES connect each hyperedge square to ALL its member vertices
- Each square is a "hub" for one hyperedge, with lines radiating out like a star
- All lines connected to the SAME SQUARE belong to that hyperedge
- To count hyperedges: count the SQUARE NODES in the inner ring
- To find members of a hyperedge: look at all vertices connected to that square

Example: If square 'e1' (blue) has lines to v0, v2, v6, v9, then hyperedge e1 = {v0, v2, v6, v9}
This is similar to bipartite_radial but emphasizes the star-like connection pattern.""",
    
    'annotated_network': """In this annotated network visualization:
- VERTICES are shown as black circles with white labels (v0, v1, v2, etc.)
- EDGES are drawn as colored lines between vertices
- Each edge has a SMALL TEXT LABEL showing which hyperedge(s) it belongs to
- The label format is "eX" or "eX,eY" if the edge belongs to multiple hyperedges
- To find all members of hyperedge eX: look for all edges labeled with 'eX' and collect connected vertices
- To count hyperedges: identify all unique 'eX' labels mentioned in the edge annotations

Example: If you see:
- Edge v1-v2 labeled 'e0'
- Edge v1-v3 labeled 'e0'
- Edge v2-v3 labeled 'e0'
Then hyperedge e0 = {v1, v2, v3} (all vertices connected by edges labeled 'e0')

Note: This requires carefully reading edge labels to reconstruct hyperedges.""",
    
    'networkx': """In this NetworkX-style hypergraph visualization:
- VERTICES are shown as BLACK CIRCLES with white labels (v0, v1, v2, etc.)
- HYPEREDGES are represented by MULTIPLE CONCENTRIC COLORED OUTLINES (like nested ellipses or polygons)
- Each hyperedge has a unique color, and multiple outline layers of that color surround all vertices in the hyperedge
- The outlines form convex hulls around the member vertices
- A LEGEND on the right side shows the color-to-hyperedge mapping (e.g., red = e0, blue = e1)
- To count hyperedges: count the number of entries in the legend OR count different colored outline groups
- To find members of a hyperedge: locate the colored outlines corresponding to that hyperedge and identify which vertices are enclosed

Example: If the legend shows a blue patch labeled "e2", find the blue nested outlines in the graph and identify which vertices (v0, v4, v7, etc.) are surrounded by those blue outlines. Those vertices form hyperedge e2.
Note: You need to visually trace the colored regions to determine hyperedge membership.""",
}


def get_instruction(layout: str) -> str:
    """获取指定布局的可视化说明"""
    if layout in VISUALIZATION_INSTRUCTIONS:
        return VISUALIZATION_INSTRUCTIONS[layout]
    
    # 默认返回bipartite_horizontal说明（最有效）
    return VISUALIZATION_INSTRUCTIONS['bipartite_horizontal']


def get_short_instruction(layout: str) -> str:
    """获取简短的可视化说明"""
    short_instructions = {
        'shell': "VERTICES = BLACK CIRCLES (inner ring), HYPEREDGES = COLORED SQUARES (outer ring). Lines connect squares to their member vertices.",
        'bipartite_horizontal': "VERTICES = BLACK CIRCLES (top), HYPEREDGES = COLORED SQUARES (bottom). Same-colored lines connect a square to its member vertices.",
        'star_expansion': "VERTICES = BLACK CIRCLES (outer), HYPEREDGES = COLORED SQUARES (inner). Each square connects to its members like a star.",
        'annotated_network': "VERTICES = black circles. EDGES have text labels showing which hyperedge(s) they belong to.",
        'networkx': "VERTICES = black circles. HYPEREDGES = colored nested outlines + LEGEND (color → eX mapping).",
    }
    
    return short_instructions.get(layout, short_instructions['bipartite_horizontal'])


def get_visualization_mode(layout: str) -> str:
    """判断可视化模式"""
    if layout in ['bipartite_horizontal', 'shell', 'star_expansion']:
        return 'explicit_nodes'  # 超边是显式节点
    elif layout in ['annotated_network']:
        return 'annotated'  # 边标注模式
    elif layout in ['networkx']:
        return 'native'  # 原生凸包区域
    else:
        return 'explicit_nodes'  # 默认


def format_vision_question(original_question: str, layout: str, use_detailed: bool = True) -> str:
    """格式化vision问题"""
    if use_detailed:
        instruction = get_instruction(layout)
    else:
        instruction = get_short_instruction(layout)
    
    return f"{instruction}\n\nQ: {original_question}"


__all__ = [
    'VISUALIZATION_INSTRUCTIONS',
    'get_instruction',
    'get_short_instruction',
    'get_visualization_mode',
    'format_vision_question'
]
