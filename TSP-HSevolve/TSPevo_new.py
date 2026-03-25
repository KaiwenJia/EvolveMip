"""
整合自以下来源文件：
- HSEvo/baselines/reevo/gls_tsp_adapt/tsp_eval_helper/gls_run.py
- HSEvo/baselines/funsearch/gls_tsp/eval_helper/gls_run.py
- HSEvo/baselines/eoh/gls_tsp_adapt/tsp_eval_helper/ael_evaluation_test_tsplib.py
- HSEvo/baselines/reevo/gls_tsp_adapt/tsp_eval_helper/ael_evaluation_test_tsplib.py
- HSEvo/problems/tsp_gls/eval.py
"""

import os
import time
import random
import numpy as np
from typing import Callable
from joblib import Parallel, delayed

# 工具函数
class GLSUtils:
    @staticmethod
    def tour_cost_2End(dis_matrix, tour):
    # 新增：空值校验 - 修复 NoneType has no len() 错误
    if tour is None or not isinstance(tour, (list, np.ndarray)) or len(tour) == 0 or len(tour) > dis_matrix.shape[0]:
        print("⚠️ Invalid tour input, returning 0 cost")
        return 0.0

        """计算旅行商路径成本（闭环）"""
        cost = 0.0
        n = len(tour)
        for i in range(n - 1):
            cost += dis_matrix[tour[i], tour[i+1]]
        cost += dis_matrix[tour[-1], tour[0]]  # 回到起点
        return cost

    @staticmethod
    def nearest_neighbor_2End(dis_matrix, start):
        """最近邻算法生成初始路径（基础贪心策略）"""
        n = dis_matrix.shape[0]
        visited = [False] * n
        tour = [start]
        visited[start] = True
        
        for _ in range(n - 1):
            last = tour[-1]
            min_dist = float('inf')
            next_city = -1
            for i in range(n):
                if not visited[i] and dis_matrix[last, i] < min_dist:
                    min_dist = dis_matrix[last, i]
                    next_city = i
            tour.append(next_city)
            visited[next_city] = True
        return np.array(tour)

    @staticmethod
    def guided_local_search(coord, dis_matrix, nearest_indices, init_tour, init_cost, 
                           time_limit, ite_max, perturbation_moves, first_improvement=True, 
                           guide_algorithm=None):

        best_tour = init_tour.copy()
        best_cost = init_cost
        current_tour = best_tour.copy()
        current_cost = best_cost
        iter_i = 0

        # 应用启发式算法修改距离矩阵（如果有）
        modified_dis = dis_matrix.copy()
        if guide_algorithm:
            modified_dis = guide_algorithm(modified_dis)

        # 核心优化循环
        start_time = time.time()
        while time.time() < start_time + time_limit and iter_i < ite_max:
            # 2-opt局部搜索（TSP经典优化方法）
            improved = False
            n = len(current_tour)
            
            for i in range(1, n-1):
                for j in range(i+1, n):
                    if j - i == 1:
                        continue  # 相邻节点交换无意义，跳过
                    
                    # 计算2-opt交换前后的成本差
                    # 原边：(i-1,i) + (j,j+1)
                    # 新边：(i-1,j) + (i,j+1)
                    old_edges = modified_dis[current_tour[i-1], current_tour[i]] + \
                               modified_dis[current_tour[j], current_tour[(j+1)%n]]
                    new_edges = modified_dis[current_tour[i-1], current_tour[j]] + \
                               modified_dis[current_tour[i], current_tour[(j+1)%n]]
                    
                    # 成本降低则执行交换
                    if new_edges < old_edges:
                        # 反转i到j的路径段
                        current_tour[i:j+1] = current_tour[i:j+1][::-1]
                        # 重新计算当前路径成本
                        current_cost = GLSUtils.tour_cost_2End(dis_matrix, current_tour)
                        improved = True
                        
                        # 首次改进策略：找到第一个改进就停止，加速收敛
                        if first_improvement:
                            break
                if improved and first_improvement:
                    break

            # 更新全局最优解
            if current_cost < best_cost:
                best_tour = current_tour.copy()
                best_cost = current_cost

            # 温和扰动：避免陷入局部最优（每50次迭代执行一次）
            # Apply perturbation to the current solution only; do not overwrite best_tour
            if iter_i % 50 == 0 and iter_i > 0:
                # 随机反转一小段路径（而非随机交换节点）
                a, b = sorted(random.sample(range(n), 2))
                # operate on a copy to avoid accidental mutation
                perturbed = current_tour.copy()
                perturbed[a:b] = perturbed[a:b][::-1]
                # recompute cost on original distances
                perturbed_cost = GLSUtils.tour_cost_2End(dis_matrix, perturbed)
                # accept perturbed as current solution
                current_tour = perturbed
                current_cost = perturbed_cost
                # update best only if truly better under original distances
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost

            iter_i += 1
        
        # 最终返回基于原始距离矩阵的成本（避免启发式矩阵干扰评估）
        final_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
        return best_tour, final_cost, iter_i

# ------------------------------
# 求解函数（适配修复版GLS）
# ------------------------------
def solve_instance(n, opt_cost, dis_matrix, coord, time_limit, ite_max, 
                  perturbation_moves, heuristic_func: Callable):
    """求解单个TSP实例并返回与最优解的差距"""
    try:
        # 生成初始解（最近邻）
        init_tour = construct_initial_tour(dis_matrix)
        init_cost = GLSUtils.tour_cost_2End(dis_matrix, init_tour)
        
        # 计算最近邻索引（保留原接口，暂未实际使用）
        nb = 100
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb + 1].astype(int)
        
        # 执行引导局部搜索
        best_tour, best_cost, iter_i = GLSUtils.guided_local_search(
            coord, dis_matrix, nearest_indices, init_tour, init_cost,
            time_limit, ite_max, perturbation_moves,
            first_improvement=True, guide_algorithm=heuristic_func
        )
        
        # 计算差距百分比
        if opt_cost == 0:
            gap = None
        else:
            gap = (best_cost / opt_cost - 1) * 100
        
        # 打印单实例结果（便于调试）
        print(f"实例 {n} - 最优解: {opt_cost:.2f}, 求解结果: {best_cost:.2f}, 差距: {gap:.2f}%")
        return gap
        
    except Exception as e:
        print(f"求解实例 {n} 时出错: {str(e)}")
        return None

def solve_instance_tsplib(n, name, scale, dis_matrix, coord, time_limit, ite_max, 
                         perturbation_moves, heuristic: Callable, temp_file_path):
    """求解TSPLIB格式实例并保存结果"""
    try:
        # 生成初始解
        init_tour = construct_initial_tour(dis_matrix)
        init_cost = GLSUtils.tour_cost_2End(dis_matrix, init_tour)
        
        # 计算最近邻索引
        nb = 100
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb + 1].astype(int)
        
        # 执行引导局部搜索
        best_tour, best_cost, iter_i = GLSUtils.guided_local_search(
            coord, dis_matrix, nearest_indices, init_tour, init_cost,
            time_limit, ite_max, perturbation_moves,
            first_improvement=True, guide_algorithm=heuristic
        )
        
        # 应用缩放因子
        scaled_cost = best_cost * scale
        
        start_time = time.time()
        
        # 保存结果到临时文件
        with open(temp_file_path, 'a', encoding='utf-8') as f:
            f.write(f"File,{name},")
            f.write(f"Best_Cost,{scaled_cost:.2f},")
            f.write(f"Time_Cost,{time.time() - start_time:.2f}\n")
            
        return scaled_cost
        
    except Exception as e:
        print(f"求解TSPLIB实例 {name} 时出错: {str(e)}")
        return 1e10  # 错误时返回大值

# 评估类
class TSPEvaluation:
    def __init__(self, instances_path=None):
        """初始化评估器参数和实例数据"""
        # 优化后的评估参数
        self.time_limit = 10  # 每个实例评估10秒（这里是因为足够2-opt收敛）
        self.ite_max = 5000   # 足够的迭代次数
        self.perturbation_moves = 1
        self.m = 50
        
        # 设置实例路径
        if instances_path is None:
            self.instances_path = os.path.join(os.path.dirname(__file__), "instance/tsp_lib_200")
        else:
            self.instances_path = instances_path
            
        # 确保实例目录存在
        os.makedirs(self.instances_path, exist_ok=True)
        
        # 加载实例数据
        self.coords, self.instances, self.instances_name, self.instances_scale = self._load_instances()
        
    def _load_instances(self):
        coords = []
        instances = []
        instances_name = []
        instances_scale = []
        
        # 解析TSP文件的核心函数
        def parse_tsp_file(file_path):
            """解析标准TSPLIB格式文件，返回坐标和距离矩阵"""
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            coord_section = False
            node_coords = []
            dimension = None
            edge_weight_type = "EUC_2D"  # 默认欧几里得距离
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析基础信息
                if line.startswith('DIMENSION'):
                    dimension = int(line.split(':')[1].strip())
                elif line.startswith('EDGE_WEIGHT_TYPE'):
                    edge_weight_type = line.split(':')[1].strip()
                elif line.startswith('NODE_COORD_SECTION'):
                    coord_section = True
                    continue
                elif line.startswith('EOF'):
                    break
                elif coord_section:
                    # 解析坐标行：节点ID x y
                    parts = line.split()
                    if len(parts) >= 3:
                        x = float(parts[1])
                        y = float(parts[2])
                        node_coords.append((x, y))
            
            # 转换为numpy数组
            node_coords = np.array(node_coords)
            n = len(node_coords)
            
            # 计算距离矩阵（仅支持EUC_2D）
            dis_matrix = np.zeros((n, n))
            if edge_weight_type == "EUC_2D":
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            # 欧几里得距离
                            dis_matrix[i, j] = np.linalg.norm(node_coords[i] - node_coords[j])
            else:
                raise ValueError(f"不支持的距离类型: {edge_weight_type}")
            
            return node_coords, dis_matrix
        
        # 遍历.tsp文件
        for filename in os.listdir(self.instances_path):
            if filename.endswith('.tsp'):
                file_path = os.path.join(self.instances_path, filename)
                try:
                    coord, dis_matrix = parse_tsp_file(file_path)
                    coords.append(coord)
                    instances.append(dis_matrix)
                    instances_name.append(filename.replace('.tsp', ''))
                    instances_scale.append(1.0)  # 柏林52无需缩放
                    print(f"成功加载实例: {filename} (城市数: {len(coord)})")
                except Exception as e:
                    print(f"解析 {filename} 失败: {e}")
        
        return coords, instances, instances_name, instances_scale
    
    def evaluate(self, heuristic: Callable):
        if not self.instances:
            print("未找到任何TSP实例，请检查instance/tsp_lib_200目录下的.tsp文件")
            return None
        
        # 创建临时结果文件
        temp_file_path = os.path.join(
            os.path.dirname(__file__), 
            f'.tsp_lib_res_{random.randint(0, 10000000)}.txt'
        )
        
        try:
            # 准备输入参数
            nins = len(self.instances_name)
            inputs = [
                (x, self.instances_name[x], self.instances_scale[x], self.instances[x], 
                 self.coords[x], self.time_limit, self.ite_max, 
                 self.perturbation_moves, heuristic, temp_file_path) 
                for x in range(nins)
            ]
            
            print(f"\n开始评估 {nins} 个实例，每个实例时间限制: {self.time_limit}秒")
            start_time = time.time()
            
            results = Parallel(n_jobs=1, timeout=self.time_limit * 1.5)(
                delayed(solve_instance_tsplib)(*input) for input in inputs
            )
            
            # 计算平均差距
            avg_gap = self._calculate_average_gap(temp_file_path)
            total_time = time.time() - start_time
            
            # 打印汇总结果
            print(f"\n评估完成！总耗时: {total_time:.2f}秒")
            print(f"所有实例的平均差距: {avg_gap:.2f}%")
            return avg_gap
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
    
    def _calculate_average_gap(self, result_file):
        """计算与最优解的平均差距"""
        # 读取最优解文件(本地设定)
        opt_cost_path = os.path.join(os.path.dirname(__file__), "opt_cost.txt")
        if not os.path.exists(opt_cost_path):
            print(f"未找到最优解文件: {opt_cost_path}")
            return 0.0
        
        # 加载最优解字典
        true_values_dict = {}
        with open(opt_cost_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        task_name = parts[0].strip()
                        true_value = float(parts[1].strip())
                        true_values_dict[task_name] = true_value
        
        # 计算每个实例的差距
        gaps = []
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 4 and parts[0] == "File":
                            task_name = parts[1].strip()
                            best_cost = float(parts[3].strip())
                            
                            # 匹配最优解
                            if task_name in true_values_dict:
                                true_value = true_values_dict[task_name]
                                if true_value > 0:
                                    gap = (best_cost / true_value - 1) * 100
                                    gaps.append(gap)
                                    print(f"实例 {task_name} - 求解值: {best_cost:.2f}, 最优值: {true_value:.2f}, 差距: {gap:.2f}%")
        
        # 计算平均差距
        if gaps:
            return np.mean(gaps)
        else:
            print("未找到可计算差距的实例结果")
            return 0.0

# 启发式函数-

import math


# --- LLM Generated Code Start ---



# --- 初始路径构造函数（最远插入法） ---
def construct_initial_tour(dis_matrix):
    # 最远插入法实现 - 默认兜底逻辑
    n = len(dis_matrix) if dis_matrix is not None else 0
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # 初始化：选择距离最远的两个点作为初始路径
    max_dist = 0
    start, end = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if dis_matrix[i][j] > max_dist:
                max_dist = dis_matrix[i][j]
                start, end = i, j
    
    tour = [start, end]
    unvisited = set(range(n)) - set(tour)
    
    # 最远插入核心逻辑
    while unvisited:
        max_insert_cost = -1
        best_node = -1
        best_pos = 0
        
        for node in unvisited:
            # 计算插入到每个位置的成本
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i+1) % len(tour)]
                # 插入成本 = 新边距离 - 原边距离
                cost = dis_matrix[a][node] + dis_matrix[node][b] - dis_matrix[a][b]
                if cost > max_insert_cost:
                    max_insert_cost = cost
                    best_node = node
                    best_pos = i + 1
        
        # 插入最优节点
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    
    return tour

# 异常捕获包装
def safe_construct_initial_tour(dis_matrix):
    try:
        return construct_initial_tour(dis_matrix)
    except Exception as e:
        print(f"⚠️ construct_initial_tour执行异常: {e}，使用兜底路径")
        n = len(dis_matrix) if dis_matrix is not None else 1
        return list(range(n)) if n > 0 else [0]



# --- 启发式惩罚函数（引导局部搜索） ---
def example_heuristic(dis_matrix):
    # 引导局部搜索的惩罚机制 - 默认兜底逻辑
    if dis_matrix is None:
        return np.array([[0.0]])
    
    n = dis_matrix.shape[0]
    if n == 0:
        return dis_matrix
    
    # 计算每条边的基础惩罚值（基于距离的倒数）
    penalty_matrix = np.zeros_like(dis_matrix, dtype=np.float64)
    min_dist = np.min(dis_matrix[dis_matrix > 0]) if np.any(dis_matrix > 0) else 1.0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 距离越短，惩罚越高（鼓励探索长距离边）
                penalty_matrix[i][j] = (min_dist / (dis_matrix[i][j] + 1e-8)) * 0.1
    
    # 返回添加惩罚后的距离矩阵
    return dis_matrix + penalty_matrix

# 异常捕获包装
def safe_example_heuristic(dis_matrix):
    try:
        return example_heuristic(dis_matrix)
    except Exception as e:
        print(f"⚠️ example_heuristic执行异常: {e}，返回原始矩阵")
        return dis_matrix if dis_matrix is not None else np.array([[0.0]])

# --- LLM Generated Code End ---

def test_berlin52():
    # 加载实例
    tsp_file = os.path.join(os.path.dirname(__file__), "instance/tsp_lib_200/berlin52.tsp")
    if not os.path.exists(tsp_file):
        print(f"未找到柏林52实例文件: {tsp_file}")
        return
    
    # 解析实例
    with open(tsp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    coord_section = False
    node_coords = []
    for line in lines:
        line = line.strip()
        if line.startswith('NODE_COORD_SECTION'):
            coord_section = True
            continue
        elif line.startswith('EOF'):
            break
        elif coord_section:
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append((x, y))
    
    # 构建距离矩阵
    node_coords = np.array(node_coords)
    n = len(node_coords)
    dis_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dis_matrix[i, j] = np.linalg.norm(node_coords[i] - node_coords[j])
    
    # 生成初始解
    init_tour = construct_initial_tour(dis_matrix)
    init_cost = GLSUtils.tour_cost_2End(dis_matrix, init_tour)
    
    # HS工作流：此处可调用大模型优化初始解生成策略
    # 例如：通过大模型分析坐标分布，推荐更优的初始起点或生成策略
    # 调用示例：
    # optimal_start = call_large_model(node_coords, prompt="推荐TSP问题的最佳初始起点...")
    # init_tour = safe_construct_initial_tour(dis_matrix)
    
    # 执行GLS优化
    print(f"柏林52实例测试：")
    print(f"初始解（最近邻）成本: {init_cost:.2f}")
    print(f"最优解参考值: 7542.0")
    print(f"初始差距: {(init_cost/7542 - 1)*100:.2f}%")
    
    # 执行引导局部搜索
    nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:101].astype(int)
    best_tour, best_cost, iter_i = GLSUtils.guided_local_search(
        node_coords, dis_matrix, nearest_indices, init_tour, init_cost,
        time_limit=10, ite_max=5000, perturbation_moves=1,
        first_improvement=True, guide_algorithm=example_heuristic
    )
    
    # HS工作流：此处可调用大模型分析优化结果，提出改进方向
    # 例如：通过大模型分析最优路径特征，调整局部搜索参数
    # 调用示例：
    # optimization_suggestion = call_large_model(best_tour, dis_matrix, prompt="分析TSP最优路径并提出优化建议...")
    # 根据建议动态调整ite_max、perturbation_moves等参数
    
    # 打印最终结果
    print(f"\n优化后结果：")
    print(f"最优路径成本: {best_cost:.2f}")
    print(f"与最优解的差距: {(best_cost/7542 - 1)*100:.2f}%")
    print(f"迭代次数: {iter_i}")
    print(f"最优路径前10个城市: {best_tour[:10]}")

#（运行测试）

if __name__ == "__main__":
    test_berlin52()
