import numpy as np
from itertools import combinations

# Held-Karp
# Keys in table: Use the nodes in the set to reach the point in the second tuple element. 
# Values in table: Order of the optimum tour and the total distance
def tsp_dynamic(adj_matrix : np.ndarray, first_node : int = 0, last_node : int = 0) -> tuple[tuple[int], int | float]:
    n = adj_matrix.shape[0]
    init_minimum = np.sum(adj_matrix)
    table = {}
    for i in range(n):
        table[(frozenset((i,)),i)] = ((i,), adj_matrix[first_node, i])
    nodes = frozenset(range(n)).difference((first_node, last_node))
    for s in range(2, len(nodes)+1):
        for S in combinations(nodes, s):
            set_S = frozenset(S)
            for k in S:
                minimum_distance = init_minimum
                minimum_path = None
                set_Smk = set_S.difference((k,))
                for m in set_Smk:
                    table_entry = table[(set_Smk, m)]
                    distance = table_entry[1] + adj_matrix[m,k]
                    if distance < minimum_distance:
                        minimum_distance = distance
                        minimum_path = table_entry[0]
                table[(set_S, k)] = (minimum_path + (k,), minimum_distance)
    
    minimum_distance = init_minimum
    minimum_path = None
    # Find Minimum
    for i in nodes:
        table_entry = table[(nodes, i)]
        distance = table_entry[1] + adj_matrix[i,last_node]
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_path = table_entry[0] + (last_node,)
    return (minimum_path, minimum_distance)