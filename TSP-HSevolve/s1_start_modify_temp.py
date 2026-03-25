#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精准替换TSPevo.py的两个核心算法 - 支持多线程版本
"""
import os
import sys
import re
import json
import argparse
import itertools
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import textwrap
import numpy as np
import subprocess
from openai import OpenAI

API_KEY = "sk-unJvwr6PfJVvVPfD3342B114F28546368dF11aD4768b3aEa"
BASE_URL = "https://api.ai-gaochao.cn/v1"
ORIGINAL_FILE = "D:/360MoveData/Users/Administrator/Desktop/CodeAgent/TSPevo.py"
OUTPUT_FILE = "TSPevo_modified.py"
LLM_LOG_FILE = os.path.join(
    os.path.dirname(__file__),
    "api_return",
    "output_s1",
    "tsp_codegen_gemini-2.5-flash.json",
)

TARGET_FUNC1 = "construct_initial_tour"   # 初始解生成函数名
TARGET_FUNC2 = "local_search_step"        # 单步局部搜索函数名（原2-opt已抽象到这里）

# 角色定义
gurobi_q2mc_roles = {
    "algorithm_expert": {
        "name": "Algorithm Expert",
        "description": "Focuses on algorithm design and optimization, proficient in analyzing algorithm complexity and performance bottlenecks"
    },
    "code_optimizer": {
        "name": "Code Optimizer", 
        "description": "Focuses on code performance optimization, proficient in using numpy and efficient data structures"
    },
    "tsp_specialist": {
        "name": "TSP Specialist",
        "description": "Focuses on the Traveling Salesman Problem (TSP), familiar with various heuristic and exact algorithms"
    },
    "python_master": {
        "name": "Python Master",
        "description": "Proficient in Python programming, with a focus on code quality and readability"
    },
    "performance_engineer": {
        "name": "Performance Engineer",
        "description": "Focuses on performance analysis and tuning, proficient in parallel computing and memory optimization"
    }
}

def load_tsp(filepath):
    """加载TSP数据文件"""
    try:
        return [{"name": "test_data", "dist_matrix": [[0, 1, 2], [1, 0, 3], [2, 3, 0]]}]
    except Exception as e:
        print(f"加载TSP文件失败: {e}")
        return None

def process_single_combination(combination):
    """处理单个组合：数据 + 模型 + 角色 - 调用LLM生成算法"""
    test_data, model_name, (role_key, role_info) = combination
    
    print(f"处理组合: 数据={test_data['name']}, 模型={model_name}, 角色={role_info['name']}")
    
    results = []
    
    # 处理两个函数：construct_initial_tour 和 local_search_step
    functions_to_generate = [
        {
            "name": "construct_initial_tour",
            "purpose": "初始路径生成",
            "original_code": """def construct_initial_tour(dist_matrix):
    # Simple nearest neighbor implementation
    n = len(dist_matrix)
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour"""
        },
        {
            "name": "local_search_step", 
            "purpose": "路径优化",
            "original_code": """def local_search_step(current_tour, dis_matrix, modified_dis, first_improvement=True):
    # Simple 2-opt implementation
    best_tour = list(current_tour)
    best_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
    improved = False
    n = len(best_tour)
    
    for i in range(n - 1):
        for j in range(i + 2, n):
            # Perform 2-opt swap
            new_tour = best_tour[:i+1] + best_tour[i+1:j][::-1] + best_tour[j:]
            new_cost = GLSUtils.tour_cost_2End(dis_matrix, new_tour)
            
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost
                improved = True
                if first_improvement:
                    break
        if improved and first_improvement:
            break
    
    return best_tour, best_cost, improved"""
        }
    ]
    
    for func_info in functions_to_generate:
        try:
            # 根据角色调整prompt
            func_prompt = f"""You are a {role_info['name']}. {role_info['description']}.

From your professional perspective, please optimize the following TSP algorithm implementation:

Function: {func_info['name']}
Purpose: {func_info['purpose']}

Original Code:
```python
{func_info['original_code']}
```

Strict Constraints:
- Signature: The function name and parameter list must remain identical to the original.
- Interface: Input/output formats must be preserved exactly to ensure compatibility.
- Formatting: Provide the final code within a python markdown block.
- Function signature: def {func_info['name']}({', '.join([p.split('=')[0].strip() for p in func_info['original_code'].split('(')[1].split(')')[0].split(',')])})
- Algorithm: Must be better and more efficient than the simple implementation
- Ensure all strings are properly closed and function is complete
- DO NOT truncate or cut off the function

Strategy:
- Analyze the time and space complexity of the current implementation
- Identify bottlenecks and inefficiencies
- Implement optimizations using your professional expertise
- Ensure code is syntactically correct and complete
"""

            # 调用LLM生成代码
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": func_prompt}],
                temperature=0.1,
                max_tokens=8000,
                top_p=0.95
            )
            
            raw_code = response.choices[0].message.content.strip()
            
            # 提取纯代码
            if "```python" in raw_code:
                match = re.search(r"```python\s*(.*?)\s*```", raw_code, re.DOTALL)
                if match:
                    body_code = match.group(1).strip()
                else:
                    body_code = raw_code.strip()
            else:
                body_code = raw_code.strip()
            
            # 语法校验
            try:
                compile(body_code + "\n", f"<llm_body_{role_key}_{func_info['name']}>", "exec")
                success = True
                error_msg = None
            except SyntaxError as e:
                success = False
                error_msg = str(e)
            
            result = {
                "data": test_data['name'],
                "model": model_name,
                "role": role_info['name'],
                "function": func_info['name'],
                "purpose": func_info['purpose'],
                "success": success,
                "error": error_msg,
                "raw_code": raw_code,
                "body_code": body_code,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
        except Exception as e:
            result = {
                "data": test_data['name'],
                "model": model_name,
                "role": role_info['name'],
                "function": func_info['name'],
                "purpose": func_info['purpose'],
                "success": False,
                "error": str(e),
                "raw_code": None,
                "body_code": None,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
    
    return results

def run_multithreaded_processing(test_data, model_list, role_list, max_workers=8):
    """多线程处理所有组合"""
    comb_list = list(itertools.product(test_data, model_list, role_list))
    print(f"总共需要处理 {len(comb_list)} 个组合，每个组合生成2个函数")
    
    all_results = []
    successful_results = []
    failed_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_comb = {
            executor.submit(process_single_combination, comb): comb 
            for comb in comb_list
        }
        
        # 收集结果
        for future in as_completed(future_to_comb):
            try:
                comb_results = future.result()
                all_results.extend(comb_results)
                
                # 分类结果
                for result in comb_results:
                    if result['success']:
                        successful_results.append(result)
                        print(f"✅ 成功: {result['role']} - {result['function']}")
                    else:
                        failed_results.append(result)
                        print(f"❌ 失败: {result['role']} - {result['function']} - {result['error']}")
                    
            except Exception as e:
                comb = future_to_comb[future]
                print(f"处理组合 {comb} 时出错: {e}")
    
    # 保存所有结果到JSON文件
    output_json_file = "multithread_llm_results.json"
    save_results_to_json(all_results, output_json_file)
    
    print(f"\n=== 结果统计 ===")
    print(f"总任务数: {len(all_results)} (每个角色生成2个函数)")
    print(f"成功生成: {len(successful_results)}")
    print(f"失败生成: {len(failed_results)}")
    print(f"结果已保存到: {output_json_file}")
    
    # 按函数分组统计
    construct_initial_tour_results = [r for r in all_results if r['function'] == 'construct_initial_tour']
    local_search_step_results = [r for r in all_results if r['function'] == 'local_search_step']
    
    print(f"\n=== construct_initial_tour 统计 ===")
    print(f"成功: {len([r for r in construct_initial_tour_results if r['success']])}")
    print(f"失败: {len([r for r in construct_initial_tour_results if not r['success']])}")
    
    print(f"\n=== local_search_step 统计 ===")
    print(f"成功: {len([r for r in local_search_step_results if r['success']])}")
    print(f"失败: {len([r for r in local_search_step_results if not r['success']])}")
    
    # 保存成功的代码到单独文件
    if successful_results:
        print(f"\n=== 保存成功代码 ===")
        for result in successful_results:
            if result['body_code']:
                filename = f"generated_{result['role']}_{result['function']}.py"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result['body_code'])
                print(f"  - {filename}")
    
    return all_results

def save_results_to_json(results, filename):
    """保存结果到JSON文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {filename}")
    except Exception as e:
        print(f"保存JSON文件失败: {e}")

def save_llm_response(func_name: str, func_purpose: str, raw_code: str, body_code: str):
    os.makedirs(os.path.dirname(LLM_LOG_FILE), exist_ok=True)

    record = {
        "time": datetime.now().isoformat(),
        "func_name": func_name,
        "func_purpose": func_purpose,
        "raw_code": raw_code,
        "body_code": body_code,
    }

    data = []
    if os.path.exists(LLM_LOG_FILE):
        try:
            with open(LLM_LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    data = existing
                else:
                    data = [existing]
        except Exception:
            data = []

    data.append(record)

    with open(LLM_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ================= LLM调用 & 代码规范化：生成任意TSP算法（模板 + LLM片段） =================
def _normalize_indentation(body_code: str) -> str:
    return textwrap.dedent(body_code).strip()


def _build_construct_initial_tour_body(llm_snippet: str) -> str:
    """
    使用固定模板 + LLM生成的核心片段，构造完整的construct_initial_tour函数体。
    约定：LLM片段在while循环内部执行，必须最终给出一个合法的 next_city（在unvisited中）。
    可用变量：dist_matrix, n, tour(list[int]), unvisited(set[int]), current(int)
    """
    snippet_lines = llm_snippet.splitlines()
    indented_snippet = "\n".join(
        (" " * 8 + line if line.strip() else "") for line in snippet_lines
    )

    body = f"""
n = dist_matrix.shape[0]

if n == 0:
    return np.array([], dtype=int)
if n == 1:
    return np.array([0], dtype=int)

unvisited = set(range(n))
current = 0
tour = [current]
unvisited.remove(current)

while unvisited:
{indented_snippet}
    # 兜底检查：如果LLM代码没有正确设置next_city，则退回到任意一个未访问城市
    if "next_city" not in locals() or next_city not in unvisited:
        next_city = min(unvisited)

    tour.append(next_city)
    unvisited.remove(next_city)
    current = next_city

return np.array(tour, dtype=int)
""".strip("\n")
    return body


def _build_guided_local_search_body(llm_snippet: str) -> str:
    """
    使用固定模板 + LLM生成的核心片段，构造完整的guided_local_search函数体。
    约定：LLM片段在while循环内部执行，必须生成 new_tour(np.ndarray) 和 new_cost(float)。
    可用变量：current_tour, current_cost, best_tour, best_cost, dis_matrix, modified_dis,
             time, time_limit, iter_i, ite_max, random, GLSUtils
    """
    snippet_lines = llm_snippet.splitlines()
    indented_snippet = "\n".join(
        (" " * 8 + line if line.strip() else "") for line in snippet_lines
    )

    body = f"""
best_tour = init_tour.copy()
best_cost = init_cost
current_tour = best_tour.copy()
current_cost = best_cost
iter_i = 0

modified_dis = dis_matrix.copy()
if guide_algorithm:
    modified_dis = guide_algorithm(modified_dis)

start_time = time.time()
while time.time() < start_time + time_limit and iter_i < ite_max:
    iter_i += 1
{indented_snippet}
    # 兜底：如果LLM片段未设置new_tour/new_cost，则保持当前解不变
    if "new_tour" not in locals() or "new_cost" not in locals():
        new_tour = current_tour
        new_cost = current_cost

    # 若LLM给出的解更优，则接受
    if isinstance(new_tour, np.ndarray):
        candidate_tour = new_tour
    else:
        candidate_tour = np.array(new_tour, dtype=int)
    candidate_cost = float(new_cost)

    if candidate_cost < current_cost:
        current_tour = candidate_tour
        current_cost = candidate_cost
        if current_cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_cost

final_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
return best_tour, final_cost, iter_i
""".strip("\n")
    return body


def generate_algorithm_code(func_name, func_purpose):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    last_error_msg = ""
    max_retry = 3

    for attempt in range(1, max_retry + 1):
        if func_name == TARGET_FUNC1:
            prompt = f"""You are a Python Expert specializing in High-Performance Algorithmic Design. Context: You are focusing on the following combinatorial optimization problem: TSP Initial Tour Construction.

Objective: Refine the provided code snippet to maximize computational efficiency while maintaining code readability and Pythonic elegance.

Original Code: 
```python
def construct_initial_tour(dist_matrix):
    # Simple nearest neighbor implementation
    n = len(dist_matrix)
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour
```

Strict Constraints:
- Signature: The function name and parameter list must remain identical to the original.
- Interface: Input/output formats must be preserved exactly to ensure compatibility.
- Formatting: Provide the final code within a python markdown block.
- Function signature: def construct_initial_tour(dist_matrix)
- Return: numpy array of city indices forming a complete tour
- Algorithm: Must be better than simple nearest neighbor (e.g., regret-based insertion, GRASP, farthest insertion)

Strategy:
- Analyze the time and space complexity of the current implementation.
- Identify bottlenecks (e.g., redundant calculations, inefficient data structures).
- Implement optimizations (e.g., utilizing numpy, scipy, or better pruning logic).
- Think step-by-step before providing the optimized solution.
- Ensure all strings are properly closed and function is complete.
- DO NOT truncate or cut off the function.
"""
        elif func_name == TARGET_FUNC2:
            prompt = f"""You are a Python Expert specializing in High-Performance Algorithmic Design. Context: You are focusing on the following combinatorial optimization problem: TSP Local Search Optimization.

Objective: Refine the provided code snippet to maximize computational efficiency while maintaining code readability and Pythonic elegance.

Original Code:
```python
def local_search_step(current_tour, dis_matrix, modified_dis, first_improvement=True):
    # Basic 2-opt implementation
    best_tour = current_tour.copy()
    best_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
    improved = False
    
    n = len(best_tour)
    for i in range(n - 1):
        for j in range(i + 2, n):
            # Try 2-opt swap
            new_tour = best_tour[:i+1] + best_tour[i+1:j][::-1] + best_tour[j:]
            new_cost = GLSUtils.tour_cost_2End(dis_matrix, new_tour)
            
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost
                improved = True
                
                if first_improvement:
                    break
        
        if improved and first_improvement:
            break
    
    return best_tour, best_cost, improved
```

Strict Constraints:
- Signature: The function name and parameter list must remain identical to the original.
- Interface: Input/output formats must be preserved exactly to ensure compatibility.
- Formatting: Provide the final code within a python markdown block.
- Function signature: def local_search_step(current_tour, dis_matrix, modified_dis, first_improvement=True)
- Return: (best_tour, best_cost, improved)
- Use GLSUtils.tour_cost_2End for cost calculation
- Algorithm: Must improve tour using local search (2-opt, 3-opt, etc.)

Strategy:
- Analyze the time and space complexity of the current implementation.
- Identify bottlenecks (e.g., redundant calculations, inefficient data structures).
- Implement optimizations (e.g., utilizing numpy, scipy, or better pruning logic).
- Think step-by-step before providing the optimized solution.
- Ensure all strings are properly closed and function is complete.
- DO NOT truncate or cut off the function.
"""
        else:
            prompt = ""

        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8000, 
                top_p=0.95
            )
            raw_code = response.choices[0].message.content.strip()

            # 提取纯代码
            if "```python" in raw_code:
                match = re.search(r"```python\s*(.*?)\s*```", raw_code, re.DOTALL)
                if match:
                    body_code = match.group(1).strip()
                else:
                    lines = raw_code.split('\n')
                    code_lines = []
                    in_code_block = False
                    for line in lines:
                        if line.strip().startswith('```python'):
                            in_code_block = True
                            continue
                        elif line.strip().startswith('```') and in_code_block:
                            in_code_block = False
                            continue
                        elif in_code_block:
                            code_lines.append(line)
                    body_code = '\n'.join(code_lines).strip()
            elif raw_code.startswith("```"):
                lines = raw_code.split('\n')
                if len(lines) > 1:
                    body_code = '\n'.join(lines[1:-1]).strip()
                else:
                    body_code = raw_code.strip().strip('`').strip()
            else:
                body_code = raw_code.strip()
            

            body_code = body_code.strip('`').strip()
            
            # 检查代码完整性
            if not body_code.strip():
                last_error_msg = "Empty code extracted"
                continue
                
            # 检查基本语法完整性
            if func_name == TARGET_FUNC1:
                if 'def ' not in body_code or 'return ' not in body_code:
                    last_error_msg = "Incomplete function structure"
                    continue
            elif func_name == TARGET_FUNC2:
                if 'def ' not in body_code or 'return ' not in body_code:
                    last_error_msg = "Incomplete function structure" 
                    continue

            # 语法校验
            try:
                compile(body_code + "\n", f"<llm_body_{func_name}>", "exec")
                save_llm_response(func_name, func_purpose, raw_code, body_code)
                print(f"{func_name} 代码生成成功")
                return body_code
            except SyntaxError as se:
                last_error_msg = f"SyntaxError: {se}"
                print(f"第 {attempt} 次生成的 {func_name} 代码存在语法错误，将重试：{se}")
                save_llm_response(func_name, func_purpose, raw_code, body_code)
                continue

        except Exception as e:
            last_error_msg = f"Exception during LLM call: {e}"
            print(f"第 {attempt} 次调用生成 {func_name} 代码失败：{e}")
            continue

    raise RuntimeError(f"无法生成可编译的 {func_name} 函数体，多次重试后仍有语法错误或调用失败：{last_error_msg}")


def replace_algorithm_blocks():
    """
    核心逻辑：只替换指定两个函数的体，其余代码完全保留
    """
    if not os.path.exists(ORIGINAL_FILE):
        print(f"原文件不存在：{ORIGINAL_FILE}")
        print(f"请检查路径是否正确，当前路径：{ORIGINAL_FILE}")
        return False
    
    # 读取原文件内容
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换第一个函数：初始路径生成（construct_initial_tour）
    print(f"\n替换函数：{TARGET_FUNC1}（初始路径生成）")
    func1_body = generate_algorithm_code(
        TARGET_FUNC1,
        "Generate initial TSP tour using the given distance matrix (better than simple nearest neighbor)."
    )
    func1_pattern = r"(^[ \t]*def " + re.escape(TARGET_FUNC1) + r"\(.*?\):)(.*?)(?=^[ \t]*def |^[ \t]*class |^if __name__|$)"
    func1_match = re.search(func1_pattern, content, re.DOTALL | re.MULTILINE)
    
    if func1_match:
        # 提取原函数定义行 + 缩进级别（包括在类中的缩进）
        func1_def = func1_match.group(1)
        # 行首前导空白作为函数定义所在缩进
        indent_match = re.match(r"^[ \t]*", func1_def)
        base_indent = indent_match.group(0) if indent_match else ""
        # 函数体相对def再缩进4个空格（顶层函数=4空格，类方法=8空格等）
        indent = base_indent + " " * 4
        
        # 给新函数体添加正确缩进
        func1_body_lines = func1_body.split('\n')
        func1_body_indented = []
        for line in func1_body_lines:
            if line.strip():
                func1_body_indented.append(indent + line)
            else:
                func1_body_indented.append(line)
        func1_body_indented = '\n'.join(func1_body_indented)
        
        # 拼接新函数（定义行 + 缩进后的体）
        new_func1 = func1_def + '\n' + func1_body_indented
        # 替换原函数
        content = content.replace(func1_match.group(0), new_func1)
        print(f"{TARGET_FUNC1} 替换完成")
    else:
        print(f"未找到函数 {TARGET_FUNC1}，跳过替换")
        print(f"请确认原文件中是否有该函数，或检查函数名是否正确")
    
    # 替换第二个函数：路径优化（guided_local_search）
    print(f"\n替换函数：{TARGET_FUNC2}（路径优化）")
    func2_body = generate_algorithm_code(
        TARGET_FUNC2,
        "Perform guided local search/metaheuristic optimization on a TSP tour (keep same input/output behavior)."
    )
    func2_pattern = r"(^[ \t]*def " + re.escape(TARGET_FUNC2) + r"\(.*?\):)(.*?)(?=^[ \t]*def |^[ \t]*class |^if __name__|$)"
    func2_match = re.search(func2_pattern, content, re.DOTALL | re.MULTILINE)
    
    if func2_match:
        func2_def = func2_match.group(1)
        indent_match = re.match(r"^[ \t]*", func2_def)
        base_indent = indent_match.group(0) if indent_match else ""
        indent = base_indent + " " * 4
        
        # 给新函数体添加正确缩进
        func2_body_lines = func2_body.split('\n')
        func2_body_indented = []
        for line in func2_body_lines:
            if line.strip():
                func2_body_indented.append(indent + line)
            else:
                func2_body_indented.append(line)
        func2_body_indented = '\n'.join(func2_body_indented)
        
        new_func2 = func2_def + '\n' + func2_body_indented
        content = content.replace(func2_match.group(0), new_func2)
        print(f"{TARGET_FUNC2} 替换完成")
    else:
        print(f"未找到函数 {TARGET_FUNC2}，跳过替换")
        print(f"请确认原文件中是否有该函数，或检查函数名是否正确")
    
    # 保存替换后的文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n替换完成！新文件保存为：{OUTPUT_FILE}")
        print(f"仅替换了 {TARGET_FUNC1} 和 {TARGET_FUNC2}，其余代码完全保留")
        return True
    except Exception as e:
        print(f"保存文件失败：{e}")
        return False

# ================= 运行验证 =================
def verify_modified_file():
    """验证替换后的文件是否能正常运行"""
    if not os.path.exists(OUTPUT_FILE):
        return False, "替换后的文件不存在"
    
    try:
        print("\n验证替换后的文件运行状态...")
        result = subprocess.run(
            [sys.executable, OUTPUT_FILE],
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            return True, f"运行成功！\n输出预览：\n{result.stdout[:500]}"
        else:
            error_msg = f"运行报错（返回码：{result.returncode}）\n"
            error_msg += f"错误信息：\n{result.stderr[:800]}"
            return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "运行超时（超过5分钟），文件可能存在死循环"
    except Exception as e:
        return False, f"验证异常：{str(e)}"

# ================= 主函数 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='start-scripts')
    parser.add_argument('--model_name', type=str, required=False, default="gemini-2.5-flash", help='model name, options: []')
    parser.add_argument('--prompt_name', type=str, required=False, default='conflict_check', help='prompt name, options: []')
    parser.add_argument('--solver_name', type=str, required=False, default='python',
                         help='solver name, options: []')
    parser.add_argument('--data_path', type=str, required=False, default='/data1/AHD/data/TSP/a280.tsp',
                        help='path of the test data, options: []')
    parser.add_argument('--max_workers', type=int, required=False, default=8,
                        help='maximum number of worker threads')
    parser.add_argument('--use_multithreading', action='store_true', 
                        help='enable multithreading mode')
    args = parser.parse_args() 

    model_name = args.model_name
    solver_name = args.solver_name
    prompt_name = args.prompt_name
    filepath = args.data_path
    max_workers = args.max_workers
    data_name =  filepath.split("/")[-1].split(".")[0]

    print("model name:\t", model_name)
    print("prompt name:\t", prompt_name)
    print("max workers:\t", max_workers)

    if args.use_multithreading:
        # 多线程模式
        print("\n=== 多线程模式 ===")
        loaded_data = load_tsp(filepath)
        if loaded_data:
            test_data = loaded_data[:10]
            role_list = list(gurobi_q2mc_roles.items())[:5]
            model_list = [model_name]
            
            print(f"使用 {len(test_data)} 个测试数据")
            print(f"使用 {len(role_list)} 个角色")
            print(f"使用模型: {model_name}")
            
            results = run_multithreaded_processing(test_data, model_list, role_list, max_workers)
            
            print(f"\n多线程处理完成，共处理 {len(results)} 个组合")
            for result in results:
                print(f"  - {result}")
        else:
            print("数据加载失败")
    else:
        # 原有的单线程模式
        print("\n=== 单线程模式 ===")
        replace_success = replace_algorithm_blocks()
        
        if replace_success:
            valid, result = verify_modified_file()
            if valid:
                print(f"\n{result}")
            else:
                print(f"\n{result}")
                print("\n提示：文件结构未变，仅核心算法替换，可直接打开以下文件调试：")
                print(f"   {OUTPUT_FILE}")
        else:
            print("\n替换失败，请检查原文件路径和函数名是否正确")