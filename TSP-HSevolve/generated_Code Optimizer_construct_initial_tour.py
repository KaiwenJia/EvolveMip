import numpy as np

def construct_initial_tour(dist_matrix):
    n = len(dist_matrix)
    tour = [0]  # Start at city 0
    
    # Use a boolean NumPy array to track visited cities.
    # This allows for efficient masking using NumPy's vectorized operations.
    visited = np.zeros(n, dtype=bool)
    visited[0] = True  # Mark the starting city as visited
    
    current_city = 0
    
    # The loop runs n-1 times to add the remaining n-1 cities to the tour.
    for _ in range(n - 1):
        # Get the distances from the current city to all other cities.
        # .copy() is crucial to avoid modifying the original dist_matrix row,
        # as we will temporarily set visited city distances to infinity.
        distances_from_current = dist_matrix[current_city, :].copy()
        
        # Set distances to already visited cities to infinity.
        # This ensures they are not considered when finding the minimum.
        # This is a highly efficient vectorized operation.
        distances_from_current[visited] = np.inf
        
        # Find the index of the city with the minimum distance.
        # np.argmin is a C-optimized function.
        nearest_city = np.argmin(distances_from_current)
        
        tour.append(nearest_city)
        visited[nearest_city] = True  # Mark the newly added city as visited
        current_city = nearest_city  # Move to the nearest city
    
    return tour