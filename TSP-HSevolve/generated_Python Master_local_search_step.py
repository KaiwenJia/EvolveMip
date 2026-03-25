def local_search_step(current_tour, dis_matrix, modified_dis, first_improvement=True):
    best_tour = list(current_tour)
    best_cost = GLSUtils.tour_cost_2End(dis_matrix, best_tour)
    improved = False
    n = len(best_tour)

    # A 2-opt swap requires at least 4 cities to make a non-trivial change.
    # The loops naturally handle smaller n, resulting in no swaps.
    if n < 4:
        return best_tour, best_cost, improved

    # Flag to control breaking out of outer loop
    outer_loop_break = False

    for i in range(n - 1):
        for j in range(i + 2, n):
            # Identify the four cities involved in the potential swap
            # The original 2-opt reverses the segment from index i+1 to j-1.
            # Edges broken: (best_tour[i], best_tour[i+1]) and (best_tour[j-1], best_tour[j])
            # Edges added: (best_tour[i], best_tour[j-1]) and (best_tour[i+1], best_tour[j])
            
            c1 = best_tour[i]
            c2 = best_tour[i+1]
            c3 = best_tour[j-1]
            c4 = best_tour[j]

            # Calculate the change in cost if this swap were performed
            # This avoids recalculating the entire tour cost (O(n)) for each potential swap.
            delta_cost = (dis_matrix[c1][c3] + dis_matrix[c2][c4]) - \
                         (dis_matrix[c1][c2] + dis_matrix[c3][c4])

            if delta_cost < 0:  # If the swap results in a lower cost
                # Only construct the new tour if an improvement is found
                new_tour = best_tour[:i+1] + best_tour[i+1:j][::-1] + best_tour[j:]
                
                best_tour = new_tour
                best_cost += delta_cost # Update cost incrementally
                improved = True

                if first_improvement:
                    outer_loop_break = True
                    break  # Break from inner loop
        
        if outer_loop_break:
            break  # Break from outer loop

    return best_tour, best_cost, improved