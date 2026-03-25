import numpy as np

# Placeholder for GLSUtils.tour_cost_2End.
# In a real application, this would be imported from a utility module.
# This function is called only once at the beginning to get the initial cost.
class GLSUtils:
    @staticmethod
    def tour_cost_2End(dis_matrix, tour):
        cost = 0
        n = len(tour)
        if n == 0:
            return 0
        for i in range(n):
            cost += dis_matrix[tour[i]][tour[(i + 1) % n]]
        return cost

def local_search_step(current_tour, dis_matrix, modified_dis, first_improvement=True):
    """
    Optimizes a given TSP tour using the 2-opt local search algorithm.

    This optimized version uses incremental cost calculation to avoid
    recalculating the entire tour cost for each potential swap,
    significantly improving performance.

    Args:
        current_tour (list): The current tour as a list of node indices.
        dis_matrix (list of lists or numpy.ndarray): The distance matrix
                                                     where dis_matrix[i][j]
                                                     is the distance between
                                                     node i and node j.
        modified_dis (any): This parameter is part of the original signature
                            but is not used in the 2-opt logic.
        first_improvement (bool): If True, the search stops and applies the
                                  first improving swap found. If False, it
                                  searches for the best possible 2-opt swap
                                  in the current iteration and applies it.

    Returns:
        tuple: A tuple containing:
            - best_tour (list): The optimized tour.
            - best_cost (float): The cost of the optimized tour.
            - improved (bool): True if an improvement was found and applied,
                               False otherwise.
    """
    best_tour = list(current_tour) # Create a mutable copy to work with
    n = len(best_tour)

    # A 2-opt swap requires at least 4 nodes to make a meaningful change.
    # For n < 4, any 2-opt swap effectively results in the same tour or is not possible.
    if n < 4:
        return best_tour, GLSUtils.tour_cost_2End(dis_matrix, best_tour), False

    best_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
    improved = False

    # Variables to store the best swap found if first_improvement is False
    best_i_overall = -1
    best_j_overall = -1
    best_delta_overall = 0.0 # Initialize to 0, as we are looking for negative delta

    # Flag to break out of outer loop if first_improvement is True
    found_first_improvement = False

    # Iterate through all possible 2-opt swap pairs (i, j)
    # i goes from 0 to n-2
    for i in range(n - 1):
        # j must be at least i+2 for a valid 2-opt swap
        # (i.e., there must be at least one node between i and j)
        # j goes from i+2 to n-1
        for j in range(i + 2, n):
            # Identify the four nodes involved in the potential swap
            # The edges being removed are (node_i, node_i_plus_1) and (node_j, node_j_plus_1)
            # The edges being added are (node_i, node_j) and (node_i_plus_1, node_j_plus_1)
            
            node_i = best_tour[i]
            node_i_plus_1 = best_tour[i+1]
            node_j = best_tour[j]
            node_j_plus_1 = best_tour[(j+1)%n] # Handles wrap-around for the last segment (j=n-1)

            # Calculate the cost of the old edges
            old_edges_cost = dis_matrix[node_i][node_i_plus_1] + dis_matrix[node_j][node_j_plus_1]
            
            # Calculate the cost of the new edges
            new_edges_cost = dis_matrix[node_i][node_j] + dis_matrix[node_i_plus_1][node_j_plus_1]
            
            # Calculate the change in total tour cost
            delta = new_edges_cost - old_edges_cost

            if delta < 0: # An improvement is found
                improved = True
                if first_improvement:
                    # Apply the swap immediately
                    # The segment best_tour[i+1:j] needs to be reversed
                    best_tour[i+1:j] = best_tour[i+1:j][::-1]
                    best_cost += delta
                    found_first_improvement = True
                    break # Break from inner loop, as we only need the first improvement
                else:
                    # If not first_improvement, keep track of the best delta found so far
                    if delta < best_delta_overall: # We want the most negative delta
                        best_delta_overall = delta
                        best_i_overall = i
                        best_j_overall = j
        
        if found_first_improvement:
            break # Break from outer loop if first_improvement was applied

    # If first_improvement is False and an improvement was found, apply the single best swap
    if not first_improvement and improved:
        # Apply the best swap found across all iterations
        best_tour[best_i_overall+1:best_j_overall] = best_tour[best_i_overall+1:best_j_overall][::-1]
        best_cost += best_delta_overall

    return best_tour, best_cost, improved