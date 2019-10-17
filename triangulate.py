from collections import defaultdict
from itertools import permutations

from scipy.spatial import Delaunay


def get_neighbors_dict(locations):

    tri = Delaunay(locations)

    neighbors = defaultdict(set)
    for simplex in tri.vertices:
        for i, j in permutations(simplex, 2):
            neighbors[i].add(j)

    return tri, neighbors
