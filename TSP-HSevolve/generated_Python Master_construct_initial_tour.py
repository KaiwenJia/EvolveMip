import math

def construct_initial_tour(dist_matrix):
    """
    Constructs an initial tour for the Traveling Salesperson Problem (TSP)
    using a Multiple-Start Nearest Neighbor heuristic.

    This approach runs the Nearest Neighbor algorithm starting from each possible city
    and returns the shortest tour found among them. This generally yields a
    significantly better initial tour quality compared to starting from a fixed city (e.g., city 0).

    Args:
        dist_matrix (list of lists or numpy.ndarray): A square matrix where dist_matrix[i, j]
                                                      represents the distance between city i and city j.
                                                      Assumed to be symmetric with 0s on the diagonal.
                                                      Supports tuple indexing like dist_matrix[i, j].

    Returns:
        list: A list of integers representing the sequence of cities in the initial tour.
              The tour implicitly starts and ends at the same city (by connecting the last
              city in the list back to the first).
    """
    n = len(dist_matrix)

    # Handle edge cases for 0 or 1 city
    if n == 0:
        return []
    if n == 1:
        return [0]

    best_tour = None
    min_total_distance = float('inf')

    # Helper function to calculate the total distance of a given tour
    def _calculate_tour_distance(tour):
        total_dist = 0.0
        for i in range(len(tour)):
            # Add distance from current city to next city (wrapping around for the last segment)
            total_dist += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return total_dist

    # Helper function for the single-start Nearest Neighbor algorithm
    def _nearest_neighbor_tour_from_start(start_node):
        current_tour = [start_node]
        # Use a set for O(1) average-case removal and lookup
        unvisited = set(range(n))
        unvisited.remove(start_node)
        current_city = start_node

        while unvisited:
            nearest_city = -1
            min_dist_to_nearest = float('inf')

            # Iterate through unvisited cities to find the nearest one
            for city in unvisited:
                distance = dist_matrix[current_city, city]
                if distance < min_dist_to_nearest:
                    min_dist_to_nearest = distance
                    nearest_city = city
            
            # This condition should ideally not be met if unvisited is not empty
            # and dist_matrix contains valid distances.
            if nearest_city == -1:
                break # Should not happen in a valid TSP instance

            current_tour.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        return current_tour

    # Iterate through all possible starting cities to find the best Nearest Neighbor tour
    for start_node in range(n):
        current_tour = _nearest_neighbor_tour_from_start(start_node)
        current_total_distance = _calculate_tour_distance(current_tour)

        if current_total_distance < min_total_distance:
            min_total_distance = current_total_distance
            best_tour = current_tour

    return best_tour