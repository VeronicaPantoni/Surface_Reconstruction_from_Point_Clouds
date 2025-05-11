import sys
import numpy as np
import polyscope as ps
from scipy.spatial import KDTree as kdtree
from collections import defaultdict, deque
from union_find import UnionFind 
import time
from scipy.spatial import Delaunay

def read_xyz_file(file_path: str) -> np.ndarray:
    """Reads a .xyz file and returns a numpy array of shape (N,3) where N is the number of points
    Args:
        file_path (str): pth to the .xyz file
    Returns:
        np.ndarray: points
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        values = line.strip().split()
        if len(values) == 3:
            points.append([float(x) for x in values])

    return np.array(points)

## NORMAL ESTIMATION -------------------------------------
def knn_graph(points, k):
    tree = kdtree(points) # Build the KDTree
    _, neighbors_indices = tree.query(points, k=k+1)  # k+1 because the point itself is included in the list

    return neighbors_indices[:, 1:],tree  # Exclude the point itself 


def best_fitting_plane(points, indices):
    bf_plane_points = points[indices]

    centroid = np.mean(bf_plane_points, axis=0)
    M = bf_plane_points - centroid  # Matrix distances from the centroid
    K = np.dot(M.T, M)  # Covariance matrix
    eig_val, eig_vec = np.linalg.eig(K)

    normal = eig_vec[:, np.argmin(eig_val)] # the eigenvector corresponding to the smallest eigenvalue

    return normal


def normal_estimation(knn_indices,points):

    N = points.shape[0]
    normals = np.zeros((N, 3))
    
    for i in range(N):
        indices = [i] + list(knn_indices[i])                # the points itself is included for computing the plane
        normals[i] = best_fitting_plane(points, indices)    # the normal is computed

    return normals

def kruskal_MST(knn_indices,normals,points):

    N = len(knn_indices)
    uf = UnionFind()
    edges = set()
    
    for i in range(N):
        uf.add(i)
        for neighbor_index in knn_indices[i]:
            if (i, neighbor_index) not in edges and (neighbor_index, i) not in edges:
                weight = 1 - np.abs(np.dot(normals[i], normals[neighbor_index])) # weight function
                #weight = np.linalg.norm(points[i] - points[neighbor_index])
                edges.add((i, neighbor_index, weight))

    sorted_edges = sorted(edges, key=lambda x: x[2]) # sort the edges by weight in ascending order
    minimal_spanning_tree = []
    
    edges_plot = []
    for u, v, weight in sorted_edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            minimal_spanning_tree.append((u, v, weight))
            edges_plot.append((u, v))

    ps.register_curve_network("edges", points, np.array(edges_plot), enabled=False)

    return minimal_spanning_tree

def consistent_orientation(minimal_spanning_tree, normals,points):
    adjacency_list = defaultdict(list)
    
    for u, v, _ in minimal_spanning_tree:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    # Set an arbitrary root and traverse the spanning tree
    root = minimal_spanning_tree[0][0]
    queue = deque([root]) 
    visited = set([root])
    
    while queue:
        current = queue.popleft()
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                if np.dot(-normals[current], normals[neighbor]) > np.dot(normals[current], normals[neighbor]):
                    normals[neighbor] = -normals[neighbor]
                queue.append(neighbor)
    
    # Ensure the normal of the upmost point is pointing upward
    upmost_point_index = np.argmax(points[:, 2])
    if normals[upmost_point_index][2] < 0:
        normals = -normals

    return normals

# Remove outliers -------------------------------------
def compute_local_area(point_idx, points, neighbors_idx, normals):

    center_point = points[point_idx]
    neighbor_points = points[neighbors_idx]
    normal = normals[point_idx]
    
    e1 = np.array([1.0, 0.0, 0.0]) # First basis vector
    if abs(np.dot(e1, normal)) > 0.9: # If the normal is parallel to the x-axis
        e1 = np.array([0.0, 1.0, 0.0]) # Use the y-axis as the first basis vector
    # Second basis vector
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    # Correct first basis vector to ensure orthogonality
    e1 = np.cross(e2, normal)
    e1 /= np.linalg.norm(e1)
    
    projected_points = []
    for p in neighbor_points:
        v = p - center_point
        x = np.dot(v, e1)
        y = np.dot(v, e2)
        projected_points.append([x, y])
    
    projected_points = np.array(projected_points)
    
    # Compute 2D Delaunay triangulation
    try:
        tri = Delaunay(projected_points)
        total_area = 0
        for simplex in tri.simplices:
            p1 = projected_points[simplex[0]]
            p2 = projected_points[simplex[1]]
            p3 = projected_points[simplex[2]]
            # Area using cross product in 2D
            v1 = p2 - p1
            v2 = p3 - p1
            area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
            total_area += area
        return total_area
    except:
        return float('inf')  # Return infinity for failed cases to mark as outliers

def detect_outliers(points, neighbors_indices, normals, eta = None):
    
    local_areas = []
    for i in range(len(points)):
        area = compute_local_area(i, points, neighbors_indices[i], normals)
        local_areas.append(area)
    
    local_areas = np.array(local_areas)
    
    # Compute threshold if not provided
    if eta is None:
        median_area = np.median(local_areas) # Median of local areas
        mad = np.median(np.abs(local_areas - median_area)) # Median absolute deviation
        eta = median_area + 3.5 * mad 
    
    inlier_mask = local_areas <= eta
    inlier_indices = np.where(inlier_mask)[0]
    outlier_indices = np.where(~inlier_mask)[0]
    return points[inlier_indices], inlier_indices, outlier_indices, local_areas

# Function to visualize results
def visualize_outlier_detection(points, outlier_indices, local_areas):
    ps.init()
    cloud = ps.register_point_cloud("points for local areas", points, enabled=False)
    cloud.add_scalar_quantity("local_areas", local_areas, enabled=False)
    outliers = points[outlier_indices]
    ps.register_point_cloud("outliers", outliers, enabled=False)

    return cloud

# RANSAC--------------------------------------  
def plane_generation(points):
    # Generate a plane from three points
    p1, p2, p3 = points
    vec1 = p2 - p1
    vec2 = p3 - p1
    normal_plane = np.cross(vec1, vec2)

    # Check if the points are colinear
    if np.allclose(normal_plane, 0):
        return None, None
    
    normal_plane /= np.linalg.norm(normal_plane)  # normalize the normal
    centroid = np.mean(points, axis=0)
    return normal_plane, centroid

def ransac_function(knn_indices,points, remaining_normals, remaining_points, remaining_indices, N, K, tau, eta):

    list_inliers = []
    best_plane = None
    best_centroid = None
    min_num_inliers = 0

    for _ in range(N):
        indices = np.random.choice(remaining_indices, 1, replace=False)
        nearest_indices = knn_indices[indices[0]]
        indices_sub = np.random.choice(nearest_indices, 2, replace=False)
        triplet = points[[indices[0], indices_sub[0], indices_sub[1]]]
        normal_plane, centroid = plane_generation(triplet)

        if normal_plane is None:    # colinear points
            continue

        distances = np.abs(np.dot(remaining_points - centroid, normal_plane))
        dot_products = np.abs(np.dot(remaining_normals, normal_plane))
        inliers_mask = (distances < tau) & (dot_products > eta)
        inliers_indices = np.where(inliers_mask)[0]

        largest_component = largest_connected_component(knn_indices,inliers_indices, remaining_indices)
        num_inliers = len(largest_component)

        if num_inliers > min_num_inliers and num_inliers >= K:
            min_num_inliers = num_inliers
            list_inliers = largest_component
            best_plane = normal_plane
            best_centroid = centroid

    return best_centroid, best_plane, list_inliers

         
# Iterative RANSAC for identify all planes            
def iterative_ransac(knn_indices,points, normals, N, K, tau, eta):
    planes = []  
    remaining_points = points.copy() 
    remaining_points_indices = np.arange(len(points)) 
    remaining_normals = normals.copy()

    num_fails = 0
    while len(remaining_points) >= K: 
        best_centroid, best_plane, list_inliers = ransac_function(knn_indices,points, remaining_normals, remaining_points, remaining_points_indices, N, K, tau,eta)

        if len(list_inliers) < K:
            num_fails += 1
            if num_fails > 20:
                break
        else:
            planes.append((best_centroid, best_plane, remaining_points_indices[list_inliers]))

            remaining_points = np.delete(remaining_points, list_inliers, axis=0)
            remaining_points_indices = np.delete(remaining_points_indices, list_inliers, axis=0)
            remaining_normals = np.delete(remaining_normals, list_inliers, axis=0)

    return planes

def largest_connected_component(knn_indices,inliers_indices, remaining_indices):

    considered_remaining_indices = remaining_indices[inliers_indices]
    
    uf = UnionFind(considered_remaining_indices)

    inlier_set = set(considered_remaining_indices)

    for i in considered_remaining_indices:
        for neighbor_index in knn_indices[i, 1:]:
            if neighbor_index in inlier_set:
                uf.union(i, neighbor_index)

    components = defaultdict(list)
    for index in considered_remaining_indices:
        root = uf.find(index)
        components[root].append(index)

    if components:
        largest_component_remaining = max(components.values(), key=len)
    else:
        return []
    largest_component = inliers_indices[np.isin(considered_remaining_indices, largest_component_remaining)]

    return largest_component

# Delaunay triangulation-------------------------------------------------

def align_to_normal(points, normal):

    normal = normal / np.linalg.norm(normal)  
    centroid = np.mean(points, axis=0) 

    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)
    I = np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2)) # Rotation matrix

    aligned_points = np.dot(points - centroid, R.T)

    return aligned_points

def reconstruct_surface(plane_points, normal, std_moltip):
    
    z_axis = np.array([0, 0, 1])
    y_axis = np.array([0, 1, 0])
    x_axis = np.array([1, 0, 0])

    tolerance = 1e-2
    if np.allclose(np.cross(z_axis, normal), 0, atol=tolerance): # the normal is parallel to the z-axis
        aligned_points = plane_points[:, :2]
    elif np.allclose(np.cross(y_axis, normal), 0, atol=tolerance): # the normal is parallel to the y-axis
        aligned_points = plane_points[:, [0, 2]]
    elif np.allclose(np.cross(x_axis, normal), 0, atol=tolerance): # the normal is parallel to the x-axis
        aligned_points = plane_points[:, 1:]
    else:
        aligned_points = align_to_normal(plane_points, normal)
        aligned_points = aligned_points[:, :2]

    delaunay = Delaunay(aligned_points)

    # FOR BOUNDARIES: comment from here 
    edge_lengths = np.linalg.norm(plane_points[delaunay.simplices[:, 1]] - plane_points[delaunay.simplices[:, 0]], axis=1)
    mean_length = np.mean(edge_lengths)
    std_length = np.std(edge_lengths)
    max_length = mean_length + std_moltip * std_length  # Max edges lenght

    filtered_simplices = []
    for simplex in delaunay.simplices:
        edges = [
            np.linalg.norm(plane_points[simplex[0]] - plane_points[simplex[1]]),
            np.linalg.norm(plane_points[simplex[1]] - plane_points[simplex[2]]),
            np.linalg.norm(plane_points[simplex[2]] - plane_points[simplex[0]])
        ]

        if all(edge <= max_length for edge in edges):
            filtered_simplices.append(simplex)
    # to here

    # UNCOMMENT FOR BOUNDARIES
    # distances = []
    # for i in range(len(plane_points)):
    #     dist = np.linalg.norm(plane_points - plane_points[i], axis=1)
    #     dist = dist[dist > 0]  # Exclude self-distance
    #     if len(dist) > 0:
    #         distances.append(np.min(dist))
    # median_distance = np.median(distances)
    # epsilon = median_distance * 0.2  # Adaptive epsilon based on point density

    # # Filter triangles
    # filtered_simplices = []
    # for simplex in delaunay.simplices:
    #     is_boundary = np.zeros(3, dtype=bool)
    #     for i, vertex_idx in enumerate(simplex):
    #         dist_to_boundary = np.linalg.norm(plane_points[vertex_idx].reshape(1, -1) - boundary_points, axis=1)
    #         is_boundary[i] = np.any(dist_to_boundary < epsilon)
        
        
    #     # Combined filtering criteria
    #     if ( np.sum(is_boundary) <= 2 ):  
    #         filtered_simplices.append(simplex)
    # UNCOMMENT FOR BOUNDARIES till here 

    delaunay.simplices = np.array(filtered_simplices)
    return delaunay

# Boundary detection-------------------------------------------------

def detect_boundary(points, knn_indices, boundary_threshold):
    etas = []
    for i in range(len(points)):
        points_neighbors = points[knn_indices[i]]
        center = np.mean(points_neighbors, axis=0)
        M = points_neighbors - center
        K = np.dot(M.T, M)
        eig_val, eig_vec = np.linalg.eig(K)
        eig_val = np.sort(eig_val)[::-1]
        # print('eig_val:', eig_val)
        lamda_3, lamda_2, lamda_1 = eig_val
        # print('eig_val:', eig_val)
        eta = lamda_1 / (lamda_1 + lamda_2 + lamda_3)

        etas.append(eta)

    etas = np.array(etas)

    boundary_mask = etas > boundary_threshold
    boundary_indices = np.where(boundary_mask)[0]
    non_boundary_indices = np.where(~boundary_mask)[0]
    return points[boundary_indices], boundary_indices, non_boundary_indices, etas
 
def visualize_boundaries(points, boundary_points,  etas):
    ps.init()
    cloud = ps.register_point_cloud("etas points", points, enabled=False)
    cloud.add_scalar_quantity("etas values", etas, enabled=True)
    ps.register_point_cloud("boundary points", boundary_points, enabled=False)

    return cloud

########## Main script
if __name__=="__main__":

    start_time = time.time()

    if len(sys.argv) > 1:
        input_xyz_file = sys.argv[1]
    else:
        input_xyz_file = "data/misc/bunny.xyz"

    # Parameters
    if "part" in input_xyz_file:
        N = 50 # number of iterations
        K = 150 # min number of inliers (250)
        tau = 0.1 # distance from plane
        eta = 0 # cos of the angle between normals
        near = 10 # number of neighbors
        std_moltip = 3.5 # standard deviation moltiplicator
        boundary_threshold = 0.01 # boundary threshold

        if "part2" in input_xyz_file:
            #K = 500 
            std_moltip = 6 
            eta = 0.9 
        if "part4" in input_xyz_file:
            std_moltip = 5
            #K = 300
        if "part6" in input_xyz_file:
            eta = 0.1
            tau = 0.1
            N = 150
        if "part7" in input_xyz_file:
            eta = 0.9
            tau = 0.1
            std_moltip = 7
        if "part8" in input_xyz_file:
            #K  = 150
            eta = 0.9
            tau = 0.2
        if "part9" in input_xyz_file:
            eta = 0.9
            tau = 0.2
            std_moltip = 9

    elif "cluster" in input_xyz_file or "roof" in input_xyz_file or "tree" in input_xyz_file: 
        N = 100
        K = 100
        tau = 0.25
        eta = 0.95
        near = 20
        std_moltip = 5 
        boundary_threshold = 0.01

        if "C_cluster26.xyz" in input_xyz_file:
            tau = 0.45
            eta = 0.9
            std_moltip = 4
        if "C_cluster28.xyz" in input_xyz_file:
            K = 210
            tau = 0.35
            eta = 0.85
        if "C_cluster29.xyz" in input_xyz_file:
            K = 200
            tau = 0.25
        if "C_cluster30.xyz" in input_xyz_file:
            K = 200
        if "C_cluster31.xyz" in input_xyz_file:
            N = 90
            near = 25
            K = 255
            eta = 0.9
            tau = 0.3
            std_moltip = 3.5
        if "C_cluster32.xyz" in input_xyz_file:
            # K = 30 # Little roofs
            std_moltip = 3.7 
        if "C_cluster33.xyz" in input_xyz_file:
            # K = 50 # Little roofs
            std_moltip = 3.7
            tau = 0.3
            K = 200
        if "C_cluster34.xyz" in input_xyz_file:
            near = 10
            eta = 0.95
            std_moltip = 3.7
            tau = 0.3
        if "tree_2roofs.xyz" in input_xyz_file:
            near = 25
        if "data/roofs/tree_6sides.xyz" in input_xyz_file:
            near = 10
            tau = 0.3
            eta = 0.9
        
    else:
        N = 20
        K = 100
        tau = 0.25
        eta = 0.95
        near = 10
        std_moltip = 3.5 
        boundary_threshold = 0.01

    # points
    points = read_xyz_file(input_xyz_file)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps_points = ps.register_point_cloud("raw points", points, enabled=False)

    # Tree
    knn_indices, tree = knn_graph(points, near)
    
    #chosen_point_index = 0
    #neighbors_indices = knn_indices[chosen_point_index]
    #neighbors_points = points[neighbors_indices]

    #ps.register_point_cloud("chosen point", points[chosen_point_index].reshape(1, -1), enabled=False, color=(1, 0, 0))
    #ps.register_point_cloud("chosen point neighbors", neighbors_points, enabled=False, color=(0, 1, 0))

    # normals
    normals = normal_estimation(knn_indices,points)
    ps_points.add_vector_quantity("normals_before", normals, enabled=False)

    if "bunny_outliers.xyz" in input_xyz_file:

        # Remove outliers
        cleaned_points, inlier_indices, outlier_indices, local_areas = detect_outliers(points,knn_indices, normals)
        cloud = visualize_outlier_detection(points, outlier_indices, local_areas)
        points = cleaned_points
        normals = normals[inlier_indices]
        knn_indices, tree = knn_graph(points, near)

        # minimal spanning tree
        mst = kruskal_MST(knn_indices,normals,points)

        new_ps_points = ps.register_point_cloud("points no outliers and oriented", points, enabled=True)
        normals = consistent_orientation(mst, normals,points)
        new_ps_points.add_vector_quantity("new normals", normals, enabled=False)

    else:
        # minimal spanning tree
        mst = kruskal_MST(knn_indices,normals,points)

        # consistent orientation
        normals = consistent_orientation(mst, normals,points)
        ps_points.add_vector_quantity("normals_after", normals, enabled=False)

    if "bunny" not in input_xyz_file:

        # detect boundaries
        boundary_points, boundary_indices, non_boundary_indices, distances = detect_boundary(points, knn_indices, boundary_threshold)
        cloud = visualize_boundaries(points, boundary_points, distances)

        # RANSAC
        planes = iterative_ransac(knn_indices,points, normals, N, K, tau, eta)

        # Plot planes
        for i, (centroid, normal, inliers) in enumerate(planes):
            plane_points = points[inliers]
            ps.register_point_cloud(f"Plane {i + 1}", plane_points, enabled=False)

            # Find the intersection between plane points and boundary points
            # boundary_set = set(map(tuple, boundary_points))
            # plane_set = set(map(tuple, plane_points))
            # intersection_points = np.array(list(boundary_set & plane_set))
            # if intersection_points.size > 0:
            #     ps.register_point_cloud(f"intersection plane {i + 1} points", intersection_points, enabled=False, color=(1, 0, 0))

            # Delaunay triangulation
            delaunay = reconstruct_surface(plane_points, normal, std_moltip)
            
            if delaunay.simplices.size > 0:
                ps.register_surface_mesh(f"Plane {i + 1} Mesh", plane_points, delaunay.simplices, enabled=True, edge_width=1.0)

    end_time = time.time()

    print('Execution time:', end_time - start_time)

    ps.show()


    # NOTES:
    # To run the "Boundary detection point" uncomment from line 345 to 365 and comment from 327 to 341
    # In this way the edges will be checked depending on the boundary points and not on the length of the edges