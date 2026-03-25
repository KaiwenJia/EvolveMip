import numpy as np

def construct_initial_tour(dist_matrix):
    """
    Optimized initial tour generation using the Nearest Neighbor algorithm.
    Leverages NumPy for efficient vectorized operations, significantly reducing
    the constant factor of the O(N^2) complexity compared to the pure Python implementation.

    Args:
        dist_matrix (np.ndarray): A 2D NumPy array representing the distance matrix.
                                  dist_matrix[i, j] is the distance from city i to city j.
                                  Assumes dist_matrix[i, i] = 0 and dist_matrix[i, j] >= 0.

    Returns:
        list: A list of integers representing the initial tour (sequence of city indices).
              The tour always starts at city 0.
    """
    # Ensure dist_matrix is a NumPy array. The original code's indexing
    # `dist_matrix[current, x]` implies it's already a NumPy array or
    # an object supporting tuple indexing. For performance, we proceed assuming this.
    # If dist_matrix could be a list of lists, an initial conversion
    # `dist_matrix = np.array(dist_matrix)` would be necessary, adding O(N^2) overhead once.

    n = len(dist_matrix)
    
    # Initialize the tour starting from city 0
    tour = [0]
    
    # Use a boolean NumPy array to keep track of visited cities.
    # This is highly efficient for vectorized operations (masking) compared to a Python set.
    visited = np.zeros(n, dtype=bool)
    visited[0] = True  # Mark the starting city as visited
    
    current_city = 0
    
    # The loop runs N-1 times to visit all remaining N-1 cities
    for _ in range(n - 1):
        # Get the row of distances from the current city to all other cities.
        # This is an O(N) operation, potentially a view or a copy depending on dist_matrix.
        distances_from_current = dist_matrix[current_city, :]
        
        # Create a temporary array to manipulate distances without altering the original matrix.
        # This copy is an O(N) operation.
        temp_distances = distances_from_current.copy()
        
        # Mark distances to already visited cities as infinity.
        # This ensures they are not selected by np.argmin.
        # This is an O(N) vectorized operation, highly optimized in C.
        temp_distances[visited] = np.inf
        
        # Find the index of the city with the minimum distance among unvisited cities.
        # np.argmin is an O(N) operation, implemented in C and can leverage parallelism.
        nearest_city = np.argmin(temp_distances)
        
        # Add the nearest city to the tour
        tour.append(nearest_city)
        
        # Mark the new city as visited
        visited[nearest_city] = True
        
        # Update the current city for the next iteration
        current_city = nearest_city
            
    return tour