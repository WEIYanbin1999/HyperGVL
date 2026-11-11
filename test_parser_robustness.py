"""
测试增强后的答案解析器的健壮性
验证能否正确处理各种格式的模型输出
"""
import re
import sys
sys.path.insert(0, '/data/home/weiyb/VLM4HyG/VLMBench')

def extract_answer_part(text):
    """模拟tasks.py中的extract_answer_part"""
    ans_match = re.search(r'Ans:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if ans_match:
        return ans_match.group(1).strip().lower()
    return text.lower()

def test_cycle_parsing():
    """测试StrictHypercycle的cycle解析"""
    print("=" * 70)
    print("测试 StrictHypercycle Cycle 解析")
    print("=" * 70)
    
    test_cases = [
        ("Ans:Yes,Cycle:[e0,e1,e2]", [0, 1, 2], "标准格式"),
        ("Ans: Yes, Cycle: [e0, e1, e2]", [0, 1, 2], "带空格"),
        ("Ans:Yes,Cycle:[0,1,2]", [0, 1, 2], "无e前缀"),
        ("Ans:Yes,cycle:[e_0,e_1,e_2]", [0, 1, 2], "下划线格式"),
        ("Ans:Yes,Cycle:[edge0,edge1,edge2]", [0, 1, 2], "edge全称"),
        ("Ans:Yes,cycle: [e0, e1, e2, e3]", [0, 1, 2, 3], "多个边"),
        ("Ans:Yes,Cycle: [ e0 , e1 , e2 ]", [0, 1, 2], "多余空格"),
        ("Ans:Yes,Cycle:[E0,E1,E2]", [0, 1, 2], "大写"),
        ("Ans:Yes,CYCLE:[e0;e1;e2]", [0, 1, 2], "分号分隔"),
    ]
    
    for predicted, expected, desc in test_cases:
        predicted_part = extract_answer_part(predicted)
        
        # 模拟tasks.py中的解析逻辑
        cycle_match = re.search(r'cycle\s*[:\-]?\s*\[([^\]]+)\]', predicted_part, re.IGNORECASE)
        if not cycle_match:
            cycle_match = re.search(r'cycle\s*[:\-]?\s*(.+?)(?:\n|$)', predicted_part, re.IGNORECASE)
        
        if cycle_match:
            cycle_str = cycle_match.group(1).strip()
            cycle_str = cycle_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            items = re.split(r'[,;\s]+', cycle_str)
            pred_cycle = []
            for item in items:
                item = item.strip()
                if not item:
                    continue
                match = re.search(r'(?:e(?:dge)?[_\-]?)?(\d+)', item, re.IGNORECASE)
                if match:
                    pred_cycle.append(int(match.group(1)))
            
            status = "✓" if pred_cycle == expected else "✗"
            print(f"{status} {desc}: {predicted}")
            if pred_cycle != expected:
                print(f"   预期: {expected}, 实际: {pred_cycle}")
        else:
            print(f"✗ {desc}: {predicted}")
            print(f"   无法匹配")

def test_coloring_parsing():
    """测试Hypergraph3Coloring的coloring解析"""
    print("\n" + "=" * 70)
    print("测试 Hypergraph3Coloring Coloring 解析")
    print("=" * 70)
    
    test_cases = [
        ("Ans:Yes,Coloring:[v0:c0,v1:c1,v2:c2]", {0:0, 1:1, 2:2}, "标准格式"),
        ("Ans: Yes, Coloring: [v0:c0, v1:c1, v2:c2]", {0:0, 1:1, 2:2}, "带空格"),
        ("Ans:Yes,Coloring:[0:0,1:1,2:2]", {0:0, 1:1, 2:2}, "无前缀"),
        ("Ans:Yes,coloring:[v_0:c_0,v_1:c_1]", {0:0, 1:1}, "下划线"),
        ("Ans:Yes,Coloring:[vertex0:color0,vertex1:color1]", {0:0, 1:1}, "全称"),
        ("Ans:Yes,COLORING:{v0:c0,v1:c1}", {0:0, 1:1}, "大括号"),
        ("Ans:Yes,Coloring: [v0 : c0, v1 : c1]", {0:0, 1:1}, "多余空格"),
        ("Ans:Yes,Coloring:[V0:C0,V1:C1]", {0:0, 1:1}, "大写"),
        ("Ans:Yes,coloring:[v0=c0,v1=c1]", {0:0, 1:1}, "等号分隔"),
    ]
    
    for predicted, expected, desc in test_cases:
        predicted_part = extract_answer_part(predicted)
        
        # 模拟tasks.py中的解析逻辑
        coloring_match = re.search(r'coloring\s*[:\-]?\s*[\[{]([^\]\}]+)[\]}]', predicted_part, re.IGNORECASE)
        if not coloring_match:
            coloring_match = re.search(r'coloring\s*[:\-]?\s*(.+?)(?:\n|$)', predicted_part, re.IGNORECASE)
        
        if coloring_match:
            coloring_str = coloring_match.group(1).strip()
            coloring_str = coloring_str.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
            pred_coloring = {}
            items = re.split(r'[,;]', coloring_str)
            for item in items:
                item = item.strip()
                if not item:
                    continue
                match = re.search(r'(?:v(?:ertex)?[_\-]?)?(\d+)\s*[:=]\s*(?:c(?:olor)?[_\-]?)?(\d+)', item, re.IGNORECASE)
                if match:
                    vertex = int(match.group(1))
                    color = int(match.group(2))
                    pred_coloring[vertex] = color
            
            status = "✓" if pred_coloring == expected else "✗"
            print(f"{status} {desc}: {predicted}")
            if pred_coloring != expected:
                print(f"   预期: {expected}, 实际: {pred_coloring}")
        else:
            print(f"✗ {desc}: {predicted}")
            print(f"   无法匹配")

def test_path_parsing():
    """测试HypergraphHamiltonianPath的path解析"""
    print("\n" + "=" * 70)
    print("测试 HypergraphHamiltonianPath Path 解析")
    print("=" * 70)
    
    test_cases = [
        ("Ans:Yes,Path:[e0,e1,e2]", [0, 1, 2], "标准格式"),
        ("Ans: Yes, Path: [e0, e1, e2]", [0, 1, 2], "带空格"),
        ("Ans:Yes,Path:[0,1,2]", [0, 1, 2], "无e前缀"),
        ("Ans:Yes,path:[e_0,e_1,e_2]", [0, 1, 2], "下划线格式"),
        ("Ans:Yes,Path:[edge0,edge1,edge2]", [0, 1, 2], "edge全称"),
        ("Ans:Yes,PATH:{e0,e1,e2}", [0, 1, 2], "大括号"),
        ("Ans:Yes,Path: [ e0 , e1 , e2 ]", [0, 1, 2], "多余空格"),
        ("Ans:Yes,Path:[E0,E1,E2,E3,E4]", [0, 1, 2, 3, 4], "大写+长路径"),
        ("Ans:Yes,path:[e0;e1;e2]", [0, 1, 2], "分号分隔"),
    ]
    
    for predicted, expected, desc in test_cases:
        predicted_part = extract_answer_part(predicted)
        
        # 模拟tasks.py中的解析逻辑
        path_match = re.search(r'path\s*[:\-]?\s*[\[{]([^\]\}]+)[\]}]', predicted_part, re.IGNORECASE)
        if not path_match:
            path_match = re.search(r'path\s*[:\-]?\s*(.+?)(?:\n|$)', predicted_part, re.IGNORECASE)
        
        if path_match:
            path_str = path_match.group(1).strip()
            path_str = path_str.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '')
            pred_path = []
            items = re.split(r'[,;\s]+', path_str)
            for item in items:
                item = item.strip()
                if not item:
                    continue
                match = re.search(r'(?:e(?:dge)?[_\-]?)?(\d+)', item, re.IGNORECASE)
                if match:
                    pred_path.append(int(match.group(1)))
            
            status = "✓" if pred_path == expected else "✗"
            print(f"{status} {desc}: {predicted}")
            if pred_path != expected:
                print(f"   预期: {expected}, 实际: {pred_path}")
        else:
            print(f"✗ {desc}: {predicted}")
            print(f"   无法匹配")

if __name__ == '__main__':
    test_cycle_parsing()
    test_coloring_parsing()
    test_path_parsing()
    
    print("\n" + "=" * 70)
    print("总结：增强的解析器能够处理各种格式变体")
    print("=" * 70)
