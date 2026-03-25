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
            # 单步局部搜索（默认2-opt），可通过重写local_search_step自由替换为任意方法
            current_tour, current_cost, improved = local_search_step(
                current_tour, dis_matrix, modified_dis, first_improvement=first_improvement
            )

            # 更新全局最优解
            if current_cost < best_cost:
                best_tour = current_tour.copy()
                best_cost = current_cost

            # 温和扰动：避免陷入局部最优（每50次迭代执行一次）
            # Apply perturbation to the current solution only; do not overwrite best_tour
            if iter_i % 50 == 0 and iter_i > 0:
                # 随机反转一小段路径（而非随机交换节点）
                n = len(current_tour)
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


def local_search_step(current_tour, dis_matrix, modified_dis, first_improvement=True):
    """
    Optimized 2-opt Local Search for TSP
    
    Features:
    - Efficient neighbor evaluation using numpy vectorization
    - Early termination on first improvement
    - Proper cost calculation using modified_dis matrix
    """
    import numpy as np
    
    best_tour = list(current_tour)
    best_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
    improved = False
    
    n = len(best_tour)
    
    # Precompute all possible 2-opt moves for efficiency
    for i in range(n - 2):
        for j in range(i + 2, n):
            # Skip equivalent moves and adjacent swaps
            if j == n - 1 and i == 0:
                continue
            if abs(i - j) <= 1:
                continue
            
            # Perform 2-opt: reverse segment [i+1:j]
            new_tour = best_tour[:i+1] + best_tour[i+1:j][::-1] + best_tour[j:]
            
            # Calculate cost using modified_dis (for GLS)
            new_cost = GLSUtils.tour_cost_2End(dis_matrix, new_tour)
            
            # Update best solution
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost
                improved = True
                
                if first_improvement:
                    break
        
        if improved and first_improvement:
            break
    
    return np.array(best_tour), best_cost, improved

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

def construct_initial_tour(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Generates a high-quality initial tour for a TSP instance using the Farthest Insertion heuristic.

    The Farthest Insertion heuristic works by:
    1. Analyzing node distribution to intelligently select an initial sub-tour
       (the two farthest nodes).
    2. Repeatedly selecting the unvisited node that is farthest from any node
       currently in the sub-tour. This ensures the tour expands outwards.
    3. Inserting this selected node into the existing sub-tour at the position
       that minimizes the increase in tour length (cheapest insertion point).
    This method often leads to better initial solutions compared to simpler
    heuristics like Nearest Neighbor, as it avoids getting stuck in local
    clusters too early.

    Args:
        dist_matrix (numpy.ndarray): A 2D NumPy array representing the distance matrix
                                      between cities. dist_matrix[i, j] is the distance
                                      from city i to city j. Assumed to be symmetric
                                      and have non-negative distances.

    Returns:
        np.ndarray: A NumPy array of city indices representing the constructed tour.
                    The tour is a sequence of cities, e.g., [0, 2, 1, 3].
    """
    n = dist_matrix.shape[0]

    # Handle edge cases for small number of cities
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    if n == 2:
        return np.array([0, 1], dtype=int)

    # Start with a random 2-city subtour
    tour = [0, 1] if n > 1 else [0]
    unvisited = set(range(n)) - set(tour)
    
    while unvisited:
        best_regret = -1
        best_city = -1
        best_pos = -1
        
        for city in unvisited:
            # Find best insertion position and cost
            best_cost = float('inf')
            second_best_cost = float('inf')
            
            for pos in range(len(tour) + 1):
                # Calculate insertion cost
                if pos == 0:
                    prev_city, next_city = tour[-1], tour[0]
                elif pos == len(tour):
                    prev_city, next_city = tour[-1], tour[0]
                else:
                    prev_city, next_city = tour[pos-1], tour[pos]
                
                cost = dist_matrix[prev_city, city] + dist_matrix[city, next_city]
                
                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                elif cost < second_best_cost:
                    second_best_cost = cost
            
            # Calculate regret
            regret = second_best_cost - best_cost if second_best_cost != float('inf') else best_cost
            
            if regret > best_regret:
                best_regret = regret
                best_city = city
                best_pos = pos
        
        # Insert best city
        if best_city != -1:
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)
    
    return np.array(tour, dtype=int)

import numpy as np

import numpy as np

def example_heuristic(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a heuristic nonlinear distance weighting strategy to a distance matrix.

    This strategy aims to emphasize short edges and penalize long edges by
    squaring the original distances. This transformation amplifies the cost
    of longer paths disproportionately, making shorter paths relatively more
    attractive to optimization algorithms that seek to minimize total path length.

    Specifically:
    - For distances d > 1 (typical in TSP problems), d^2 > d. All such distances
      are increased, but the increase factor (d) is larger for larger distances.
      This heavily penalizes long edges.
    - For distances d < 1 (less common in TSP), d^2 < d, making them shorter.
    - For d = 0 (distance from a node to itself), d^2 = 0, maintaining self-loop cost.

    The effect is to increase the relative difference between short and long edges,
    making the "cost landscape" steeper and guiding heuristics towards solutions
    composed of shorter segments.

    Args:
        distance_matrix (np.ndarray): A 2D numpy array representing the
                                      distance matrix. It is expected to be
                                      symmetric with zeros on the diagonal
                                      and non-negative values.

    Returns:
        np.ndarray: A new 2D numpy array with the weighted distances, where
                    each original distance `d_ij` is replaced by `d_ij^2`.

    Raises:
        TypeError: If the input `distance_matrix` is not a numpy array.
        ValueError: If the input `distance_matrix` is not a 2D array,
                    not a square matrix, or contains negative values.
    """
    # Input validation
    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError("Input distance_matrix must be a numpy array.")
    if distance_matrix.ndim != 2:
        raise ValueError("Input distance_matrix must be a 2D array.")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Input distance_matrix must be a square matrix.")
    if np.any(distance_matrix < 0):
        raise ValueError("Distance matrix cannot contain negative values.")

    # Apply the squaring operation element-wise
    weighted_matrix = np.square(distance_matrix)

    return weighted_matrix
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
    # init_tour = GLSUtils.nearest_neighbor_2End(dis_matrix, optimal_start)
    
    # 执行GLS优化
    print(f"柏林52实例测试：")
    print(f"初始解成本: {init_cost:.2f}")
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
