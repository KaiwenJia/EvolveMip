def construct_initial_tour(dist_matrix):
    n = len(dist_matrix)

    # Handle base cases for small number of cities
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Step 1: Initialize the subtour with two cities.
    # A common approach is to start with city 0 and its nearest neighbor.
    # This provides a slightly better starting point than just [0, 1].
    
    nearest_to_0 = -1
    min_dist_0 = float('inf')
    
    # Find the nearest neighbor to city 0 (excluding city 0 itself)
    for i in range(1, n):
        if dist_matrix[0, i] < min_dist_0:
            min_dist_0 = dist_matrix[0, i]
            nearest_to_0 = i
            
    # The initial tour consists of city 0 and its nearest neighbor
    tour = [0, nearest_to_0]
    
    # Keep track of unvisited cities
    unvisited = set(range(1, n))
    unvisited.remove(nearest_to_0)

    # Step 2: Iteratively insert remaining cities into the tour
    while unvisited:
        # Select an arbitrary unvisited city 'k'
        # Using set.pop() is O(1) on average and removes an arbitrary element.
        k = unvisited.pop() 
        
        min_cost_increase = float('inf')
        best_insertion_idx = -1
        
        # Find the best position to insert 'k' into the current tour
        # Iterate through all possible edges (city_i, city_j) in the current tour
        for i in range(len(tour)):
            city_i = tour[i]
            # city_j is the next city in the tour, wrapping around for the last edge
            city_j = tour[(i + 1) % len(tour)] 
            
            # Calculate the cost increase if k is inserted between city_i and city_j
            # The new edges would be (city_i, k) and (k, city_j)
            # The old edge (city_i, city_j) would be removed
            cost_increase = dist_matrix[city_i, k] + dist_matrix[k, city_j] - dist_matrix[city_i, city_j]
            
            # If this insertion yields a smaller cost increase, update the best position
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                # Insert k after city_i, which means at index (i + 1)
                best_insertion_idx = (i + 1) % len(tour) 
        
        # Insert city 'k' into the tour at the determined best position
        tour.insert(best_insertion_idx, k)
        
    return tour