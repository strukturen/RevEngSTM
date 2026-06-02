# Reverse Engineering Strut-and-Tie Models
# ================================
# Karin Yu, 2026

# Part 2: Clustering of nodes and geometry optimisation
# Geometry optimisation

__author__ = 'Karin Yu'

import numpy as np
from sklearn.cluster import DBSCAN
from src.utils import *
from shapely.geometry import Point, LineString, Polygon
from scipy import sparse
from src.solver import calcB
import cyipopt
import jax
import jax.numpy as jax_np
from jax.experimental import sparse

# JAX settings
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

def calcB_jax(Nd, Cn, dof):
    # CalcB for jax_np
    m, n1, n2 = len(Cn), Cn[:,0].astype(int), Cn[:,1].astype(int)
    l_org, X, Y = Cn[:,2], Nd[n2,0]-Nd[n1,0], Nd[n2,1]-Nd[n1,1]
    l = jax_np.nan_to_num(l_org)

    indices = jax_np.array([n1 * 2, n1 * 2 + 1, n2 * 2, n2 * 2 + 1])
    d_values = jax_np.take(dof, indices)
    d0, d1, d2, d3 = d_values

    s = jax_np.concatenate((-X/l*d0, -Y/l*d1, X/l*d2, Y/l*d3))
    r = jax_np.concatenate((n1*2, n1*2+1, n2*2, n2*2+1))
    c = jax_np.concatenate([jax_np.arange(m)]*4)
    indices = jax_np.stack([r,c], axis = 1)
    return sparse.BCOO((s,indices), shape=(len(Nd)*2, m))

def intersection_point_jax(L1_start, L1_end, L2_start, L2_end):
    # L1 expressed as: L1_start + t * (L1_end - L1_start)
    # L2 expressed as: L2_start + u * (L2_end - L2_start)

    # Directions of the lines
    dL1 = L1_end - L1_start  # Vector AB
    dL2 = L2_end - L2_start  # Vector CD

    # Calculate the determinant
    det = dL1[0] * dL2[1] - dL1[1] * dL2[0]
    eps = 1e-9

    intersection_point_default = jax_np.array([0.0, 0.0])

    # Solve for parameters t and u which determines the intersection
    t = ((L2_start[0] - L1_start[0]) * dL2[1] - (L2_start[1] - L1_start[1]) * dL2[0]) / (det+eps)
    u = ((L2_start[0] - L1_start[0]) * dL1[1] - (L2_start[1] - L1_start[1]) * dL1[0]) / (det+eps)

    # Point of intersection
    intersection_point = L1_start + t * dL1

    # Check if the intersection point is on both segments (0 ≤ t ≤ 1 and 0 ≤ u ≤ 1)
    on_segment = jax_np.logical_and((0 <= t) & (t <= 1), (0 <= u) & (u <= 1))

    return jax_np.where(on_segment, intersection_point, intersection_point_default)

def intersecting_members(Nd, CnC, CnS, dist_tol = 5, check_touching = False):
    """
    Using greedy algorithm we try to detect the intersection nodes of all members
    """
    # create list of all members
    # (X, Y), Type (-1 = C and 1 = S), index
    members = []
    for i in range(len(CnC)):
        members.append([LineString([Nd[int(CnC[i,0])],Nd[int(CnC[i,1])]]), -1, i])
    for i in range(len(CnS)):
        members.append([LineString([Nd[int(CnS[i,0])],Nd[int(CnS[i,1])]]), 1, i])
    # get pairs of intersecting members
    intersecting_members = []
    for i in range(len(members)):
        e1 = members[i][0] # LineString
        for j in range(len(members)):
            e2 = members[j][0] # LineString
            if e1.intersects(e2) and not e1.boundary.intersects(e2.boundary):
                # get intersection point
                intersection = e1.intersection(e2)
                intersecting_members.append([members[i], members[j], intersection])
    print('# Number of intersecting members:', len(intersecting_members))
    # modify Nd, CnC and CnS
    Nd_new = Nd.copy()
    CnC_new = CnC.copy()
    CnS_new = CnS.copy()
    for pair in intersecting_members:
        e1, e2, intersection = pair
        if isinstance(intersection, Point):
            # check if intersection point is already in Nd
            if intersection not in Nd_new[:, :2]:
                # add intersection point to Nd
                Nd_new = np.vstack((Nd_new, [intersection.x, intersection.y]))
                new_node_index = Nd_new.shape[0] - 1
            else:
                # find index
                new_node_index = np.where(Nd_new == [intersection.x, intersection.y])
            if e1[1] == -1:
                # e1 is a concrete strut
                CnC_new = extend_members(Nd_new, CnC_new, e1[2], new_node_index, True)
            else:
                # e1 is a steel tie
                CnS_new = extend_members(Nd_new, CnS_new, e1[2], new_node_index, False)
            if e2[1] == -1:
                # e1 is a concrete strut
                CnC_new = extend_members(Nd_new, CnC_new, e2[2], new_node_index, True)
            else:
                # e1 is a steel tie
                CnS_new = extend_members(Nd_new, CnS_new, e2[2], new_node_index, False)
    # also add touching points
    cnt_touching = 0
    if check_touching:
        members = []
        for i in range(len(CnC_new)):
            members.append([LineString([Nd_new[int(CnC_new[i, 0])], Nd_new[int(CnC_new[i, 1])]]), -1, i])
        for i in range(len(CnS_new)):
            members.append([LineString([Nd_new[int(CnS_new[i, 0])], Nd_new[int(CnS_new[i, 1])]]), 1, i])
        for n in range(Nd_new.shape[0]):
            for e in members:
                if e[0].distance(Point(Nd_new[n, :2])) <= dist_tol and not Point(Nd_new[n, :2]).touches(e[0]):
                    cnt_touching += 1
                    # add node to member
                    if e[1] == -1:
                        # e is a concrete strut
                        CnC_new = extend_members(Nd_new, CnC_new, e[2], n, True)
                    else:
                        # e is a steel tie
                        CnS_new = extend_members(Nd_new, CnS_new, e[2], n, False)
        print('# Number of touching members and nodes:', cnt_touching)
    return Nd_new, CnC_new, CnS_new, cnt_touching+len(intersecting_members)

def extend_members(Nd, Cn, index, new_node_index, is_concrete=True):
    # Cn = [n1, n2, length, area_compression, area_tension (if steel)]
    # existing member becomes n1 - new node, new member becomes new node - n2
    temp = int(Cn[index, 1].copy()) # existing second node
    Cn[index, 1] = new_node_index
    # adapt length of existing member
    Cn[index,2] = np.sqrt((Nd[int(Cn[index,0]), 0] - Nd[new_node_index, 0]) ** 2 + (Nd[int(Cn[index,0]), 1] - Nd[new_node_index, 1]) ** 2)
    # create new member
    dx, dy = Nd[temp, 0] - Nd[new_node_index, 0], Nd[temp, 1] - Nd[new_node_index, 1]
    if is_concrete:
        Cn_new = np.vstack((Cn, [new_node_index, temp, np.sqrt(dx ** 2 + dy ** 2), Cn[index, 3]]))
    else:
        Cn_new = np.vstack((Cn, [new_node_index, temp, np.sqrt(dx ** 2 + dy ** 2), Cn[index, 3], Cn[index, 4]]))
    return Cn_new

def clusterNodes(Nd, CnC, CnS,  bc_coords, eps=10, max_nodes = 5000):
    # Cluster nodes based on their coordinates using KMeans or DBSCAN.
    # Nd: mode coordinates, CnC: concrete struts, CnS : steel ties
    # returns clustered node indices
    # create a combined array of only active nodes
    Nd_combined = []
    all_nodes = []
    node_dict = dict()
    # -1 = CnC
    # 1 = CnS
    # Entry of Nd_combined = 1st Node coords, 2nd Node coords, Sign for CnC or CnS and index of CnC or CnS
    for i in range(len(CnC)):
        for j in range(2):
            if [Nd[int(CnC[i,j])][0], Nd[int(CnC[i,j])][1]] in all_nodes:
                node_dict[2*i+j] = all_nodes.index([Nd[int(CnC[i,j])][0], Nd[int(CnC[i,j])][1]])
            else:
                node_dict[2*i+j] = len(all_nodes)
                all_nodes.append([Nd[int(CnC[i,j])][0], Nd[int(CnC[i,j])][1]])
            Nd_combined.append((Nd[int(CnC[i,j])][0], Nd[int(CnC[i,j])][1], -1, i))
    for i in range(len(CnS)):
        for j in range(2):
            if [Nd[int(CnS[i,j])][0], Nd[int(CnS[i,j])][1]] in all_nodes:
                node_dict[2*i+j+2*len(CnC)] = all_nodes.index([Nd[int(CnS[i,j])][0], Nd[int(CnS[i,j])][1]])
            else:
                node_dict[2*i+j+2*len(CnC)] = len(all_nodes)
                all_nodes.append([Nd[int(CnS[i,j])][0], Nd[int(CnS[i,j])][1]])
            Nd_combined.append((Nd[int(CnS[i,j])][0], Nd[int(CnS[i,j])][1], 1, i))
    Nd_combined = np.array(Nd_combined)
    all_nodes = np.array(all_nodes)
    # check if nodes are part of boundary conditions
    bc_fixed = [] # indices of Nd_combined of BC
    bc_coords_fixed = [] # coordinates of BC
    # if so set dx_fixed or dy_fixed to 1
    for i in range(Nd_combined.shape[0]):
        if [round(Nd_combined[i, 0],2), round(Nd_combined[i, 1],2)] in bc_coords.tolist():
            bc_fixed.append(i)
            bc_coords_fixed.append([round(Nd_combined[i, 0],2), round(Nd_combined[i, 1],2)])
    # create node dictionary to reduce size of clustering
    # Use only x and y coordinates for clustering
    if all_nodes.shape[0] < max_nodes:
        print('Less than 5,000 nodes, clustering in one step:' , all_nodes.shape[0])
        coords = all_nodes  # Use only x and y coordinates
        cluster = DBSCAN(eps = eps, min_samples = 1).fit(coords)
        # get centroids
        centroids = dict()
        for label in set(cluster.labels_):
            cluster_points = coords[cluster.labels_ == label]
            centroids[label]=cluster_points.mean(axis = 0) # average of the node with respect to number of neighbors
        cluster_labels = cluster.labels_
    else:
        num_grids = int(all_nodes.shape[0] // max_nodes + 1)
        print('Minimum grids for clustering:', num_grids, 'of total nodes:', all_nodes.shape[0])
        x_min, x_max = all_nodes[:,0].min(), all_nodes[:,0].max()
        y_min, y_max = all_nodes[:,1].min(), all_nodes[:,1].max()
        xy_ratio = (x_max - x_min) / (y_max - y_min)
        y_factor = 2
        while y_factor * int(xy_ratio*y_factor) < num_grids:
            y_factor += 1
        x_factor = int(xy_ratio*y_factor)
        print('Grid factors for clustering: x - %d and y - %d' % (x_factor, y_factor))
        x_grid = np.linspace(x_min, x_max, x_factor+1)
        y_grid = np.linspace(y_min, y_max, y_factor+1)
        x_grid[-1] += 1
        y_grid[-1] += 1
        print('X grid:', x_grid)
        print('Y grid:', y_grid)
        num_clusters = 0
        centroids = dict()
        cluster_labels = np.full(all_nodes.shape[0], -1) # initialize with -1
        for i in range(x_factor):
            for j in range(y_factor):
                # get nodes in grid
                x_start, x_end = x_grid[i], x_grid[i+1]
                y_start, y_end = y_grid[j], y_grid[j+1]
                indices = np.where((all_nodes[:,0] >= x_start) & (all_nodes[:,0] < x_end) & (all_nodes[:,1] >= y_start) & (all_nodes[:,1] < y_end))[0]
                if len(indices) > 0:
                    # take subset inside grid
                    Nd_subset = all_nodes[indices]
                    coords = Nd_subset[:, :2]
                    cluster = DBSCAN(eps=eps, min_samples=1).fit(coords)
                    # get centroids
                    for label in set(cluster.labels_):
                        cluster_points = coords[cluster.labels_ == label]
                        centroids[label+num_clusters] = cluster_points.mean(axis=0)
                    cluster_labels[indices] = cluster.labels_ + num_clusters
                    num_clusters += len(set(cluster.labels_))
    # aggregate CnC and CnS
    # Nd_new = new clustered nodes, however with boundaries
    # Nd_new = [x, y, dx_fixed, dy_fixed], if dx_fixed = 1 else = 0
    # CnC_new = new clustered concrete struts
    # CnS_new = new clustered steel ties
    CnC_new = []
    CnS_new = []
    members_set = set()
    add_clusters = 0
    num_clusters = np.unique(cluster_labels).shape[0]
    # Check if two fixed nodes are in the same cluster
    for cluster_id in range(num_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        num_fixed = 0
        list_fixed_coords = list()
        list_fixed = []
        for i in range(Nd_combined.shape[0]):
            if node_dict[i] in indices:
                if i in bc_fixed: # checks boundary conditions
                    bc_ind = bc_fixed.index(i)
                    if not tuple(bc_coords_fixed[bc_ind]) in list_fixed_coords:
                        num_fixed += 1
                        list_fixed_coords.append(tuple(bc_coords_fixed[bc_ind]))
                        list_fixed.append([i])
                    else:
                        list_ind = list_fixed_coords.index(tuple(bc_coords_fixed[bc_ind]))
                        list_fixed[list_ind].append(i)
                if Nd_combined[i, 2] == 1: # is reinforcement
                    if i % 2 == 1:
                        dx_original, dy_original = Nd_combined[i, 0] - Nd_combined[i-1, 0], Nd_combined[
                            i, 1] - Nd_combined[i-1, 1]
                    else:
                        dx_original, dy_original = Nd_combined[i+1, 0] - Nd_combined[i, 0], Nd_combined[
                            i+1, 1] - Nd_combined[i, 1]
                    # CnS Tie, force boundaries
                    angle = boundAngle(getAngle(dx_original, dy_original), factor_max_angle=1)
                    # tie is horizontal
                    if abs(angle) <= 2 / 180 * np.pi or abs(np.pi - angle) <= 2 / 180 * np.pi:
                        if any([True for x,y in list_fixed_coords if y == round(Nd_combined[i, 1],2)]):
                            list_ind = [idx for idx, val in enumerate(list_fixed_coords) if val[1] == round(Nd_combined[i, 1],2)]
                            for li in list_ind:
                                list_fixed[li].append(i)
                        else:
                            num_fixed += 1
                            list_fixed_coords.append((round(Nd_combined[i, 0],2), round(Nd_combined[i, 1],2)))
                            list_fixed.append([i])
                    # tie is vertical
                    elif abs(np.pi / 2 - angle) <= 2 / 180 * np.pi:
                        if any([True for x,y in list_fixed_coords if x == round(Nd_combined[i, 0],2)]):
                            list_ind = [idx for idx, val in enumerate(list_fixed_coords) if val[0] == round(Nd_combined[i, 0],2)]
                            for li in list_ind:
                                list_fixed[li].append(i)
                        else:
                            num_fixed += 1
                            list_fixed_coords.append((round(Nd_combined[i, 0],2), round(Nd_combined[i, 1],2)))
                            list_fixed.append([i])
        if num_fixed > 1: # two fixed nodes in one cluster
            for j in range(1, num_fixed):
                print('##### Created additional cluster')
                # create new cluster
                add_clusters += 1
                for k in range(len(list_fixed[j])):
                    cluster_labels[node_dict[list_fixed[j][k]]] = num_clusters + add_clusters - 1
                centroids[num_clusters + add_clusters - 1] = list(list_fixed_coords[j])
    Nd_new = [[] for _ in range(num_clusters + add_clusters)]
    for i in range(int(Nd_combined.shape[0] / 2)):
        # get cluster labels
        # 2*i as always two are the same member
        label1 = int(cluster_labels[node_dict[2 * i]])
        label2 = int(cluster_labels[node_dict[2 * i + 1]])
        centroid1 = centroids[label1]
        centroid2 = centroids[label2]
        dx = (centroid2[0] - centroid1[0])
        dy = (centroid2[1] - centroid1[1])
        if Nd_combined[2 * i, 2] == -1:
            # CnC Strut
            # first node
            if len(Nd_new[label1]) == 0 and not 2 * i in bc_fixed:  # check if node exists
                Nd_new[label1] = [centroid1[0], centroid1[1], 0, 0]
            elif 2 * i in bc_fixed:  # fixed
                bc_index = bc_fixed.index(2 * i)
                Nd_new[label1] = [bc_coords_fixed[bc_index][0], bc_coords_fixed[bc_index][1], 1, 1]
            # second node
            if len(Nd_new[label2]) == 0 and not 2 * i + 1 in bc_fixed:  # check if node exists
                Nd_new[label2] = [centroid2[0], centroid2[1], 0, 0]
            elif 2 * i + 1 in bc_fixed:  # fixed
                bc_index = bc_fixed.index(2 * i + 1)
                Nd_new[label2] = [bc_coords_fixed[bc_index][0], bc_coords_fixed[bc_index][1], 1, 1]
            # add strut if it has not yet been added to either CnS or CnC
            if sum([1 for c in CnC_new if c[:2] == [label1, label2] or c[:2] == [label2, label1]] + [1 for s in CnS_new if s[:2] == [label1, label2] or s[:2] == [label2, label1]]) == 0:
                if np.sqrt(dx ** 2 + dy ** 2) >= 1e-5 and not tuple(
                        sorted((label1, label2))) in members_set:  # avoid zero length
                    CnC_new.append([label1, label2, np.sqrt(dx ** 2 + dy ** 2), CnC[int(Nd_combined[2 * i, 3]), 3]])
                    members_set.add((tuple(sorted((label1, label2)))))
        if Nd_combined[2 * i, 2] == 1:
            dx_original, dy_original = Nd_combined[2 * i + 1, 0] - Nd_combined[2 * i, 0], Nd_combined[2 * i + 1, 1] - Nd_combined[2 * i, 1]
            # CnS Tie, force boundaries
            angle = boundAngle(getAngle(dx_original, dy_original), factor_max_angle=1)
            # first node
            if len(Nd_new[label1]) == 0:  # check if node exists
                if 2 * i in bc_fixed:  # fixed node
                    bc_index = bc_fixed.index(2 * i)
                    Nd_new[label1] = [bc_coords_fixed[bc_index][0], bc_coords_fixed[bc_index][1], 1, 1]
                elif abs(angle) <= 2 / 180 * np.pi or abs(np.pi - angle) <= 2 / 180 * np.pi:  # horizontal bar, fix y location
                    Nd_new[label1] = [centroid1[0], Nd_combined[2 * i, 1], 0, 1]
                elif abs(np.pi / 2 - angle) <= 2 / 180 * np.pi:  # vertical bar, fix x location
                    Nd_new[label1] = [Nd_combined[2 * i, 0], centroid1[1], 1, 0]
                else:
                    Nd_new[label1] = [centroid1[0], centroid1[1], 1, 1]
            else:
                if 2 * i in bc_fixed:  # fixed node
                    bc_index = bc_fixed.index(2 * i)
                    Nd_new[label1] = [bc_coords_fixed[bc_index][0], bc_coords_fixed[bc_index][1], 1, 1]
                if Nd_new[label1][2:4] == [1, 1]:
                    pass
                elif abs(angle) <= 2 / 180 * np.pi or abs(np.pi - angle) <= 2 / 180 * np.pi:  # horizontal bar, fix y location
                    Nd_new[label1][1] = Nd_combined[2 * i, 1]
                    Nd_new[label1][3] = 1
                elif abs(np.pi / 2 - angle) <= 2 / 180 * np.pi:  # vertical bar, fix x location
                    Nd_new[label1][0] = Nd_combined[2 * i, 0]
                    Nd_new[label1][2] = 1
                else:  # diagonal bar
                    Nd_new[label1] = [Nd_combined[2 * i, 0], Nd_combined[2 * i, 1], 1, 1]
            # second node
            if len(Nd_new[label2]) == 0:  # check if node exists
                if 2 * i + 1 in bc_fixed:  # fixed node
                    bc_index = bc_fixed.index(2 * i + 1)
                    Nd_new[label2] = [bc_coords_fixed[bc_index][0], bc_coords_fixed[bc_index][1], 1, 1]
                elif abs(angle) <= 2 / 180 * np.pi or abs(np.pi - angle) <= 2 / 180 * np.pi:  # horizontal bar, fix y location
                    Nd_new[label2] = [centroid2[0], Nd_combined[2 * i+1, 1], 0, 1]
                elif abs(np.pi / 2 - angle) <= 2 / 180 * np.pi:  # vertical bar, fix x location
                    Nd_new[label2] = [Nd_combined[2 * i+1, 0], centroid2[1], 1, 0]
                else:
                    Nd_new[label2] = [centroid2[0], centroid2[1], 1, 1]
            else:
                if 2 * i + 1 in bc_fixed:  # fixed node
                    bc_index = bc_fixed.index(2 * i + 1)
                    Nd_new[label2] = [bc_coords_fixed[bc_index][0], bc_coords_fixed[bc_index][1], 1, 1]
                if Nd_new[label2][2:4] == [1, 1]:
                    pass
                elif abs(angle) <= 2 / 180 * np.pi or abs(
                        np.pi - angle) <= 2 / 180 * np.pi:  # horizontal bar, fix y location
                    Nd_new[label2][1] = Nd_combined[2 * i + 1, 1]
                    Nd_new[label2][3] = 1
                elif abs(np.pi / 2 - angle) <= 2 / 180 * np.pi:  # vertical bar, fix x location
                    Nd_new[label2][0] = Nd_combined[2 * i + 1, 0]
                    Nd_new[label2][2] = 1
                else:  # diagonal bar
                    Nd_new[label2] = [Nd_combined[2 * i + 1, 0], Nd_combined[2 * i + 1, 1], 1, 1]
            # add tie if it has not yet been added
            if sum([1 for s in CnS_new if s[:2] == [label1, label2] or s[:2] == [label2, label1]] + [1 for c in CnC_new if c[:2] == [label1,label2] or c[:2] == [label2,label1]]) == 0:
                if np.sqrt(dx ** 2 + dy ** 2) >= 1e-5 and not tuple(sorted((label1, label2))) in members_set:  # avoid zero length
                    CnS_new.append([label1, label2, np.sqrt(dx ** 2 + dy ** 2), CnS[int(Nd_combined[2 * i, 3]), 3], CnS[int(Nd_combined[2 * i, 3]), 4]])
                    members_set.add((tuple(sorted((label1, label2)))))
            elif sum([1 for c in CnC_new if c[:2] == [label1, label2] or c[:2] == [label2, label1]]) > 0:
                # replace strut by tie
                s_idx = [i for i in range(len(CnC_new)) if CnC_new[i][:2] == [label1, label2] or CnC_new[i][:2] == [label2, label1]][0]
                s_strut = CnC_new.pop(s_idx)
                CnS_new.append([label1, label2, s_strut[2], CnS[int(Nd_combined[2 * i, 3]), 3], CnS[int(Nd_combined[2 * i, 3]), 4]])
    return np.array(Nd_new), np.array(CnC_new), np.array(CnS_new)

def optimiseGeometry(Nd, CnC, CnS, f, dof, sC, sS, polygon_in = [], polygon_out = [], delta_dist = 50, scaling = 1, geom_constraint_factor=1):
    # Function variables = [Lf, X+Y, -Fc, Fs]
    # area
    a_C = jax_np.array([col[3] for col in CnC])  # negative, concrete
    a_S_C = jax_np.array([col[3] for col in CnS])  # negative, steel
    a_S_T = jax_np.array([col[4] for col in CnS])  # positive, steel
    # turn polygon_in and polygon_out into jax_np arrays
    polygon_in_jax = jax_np.array(polygon_in.exterior.coords)
    polygon_out_jax = [jax_np.array(poly.exterior.coords) for poly in polygon_out]
    # X and Y boundary
    lB_coord = []
    uB_coord = []
    direc = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, down, up
    for i in range(Nd.shape[0]):
        p = Point(Nd[i,0], Nd[i,1])
        if p.distance(polygon_in.boundary) < delta_dist or any([p.distance(poly.boundary) < delta_dist for poly in polygon_out]):
            if p.within(polygon_in) and not all([p.within(poly) for poly in polygon_out]):
                # inside geometry
                # check all sides
                delta_dist_node = []
                for d in direc:
                    p_boundary = Point(p.x + d[0]*delta_dist, p.y + d[1]*delta_dist)
                    min_dist = min(p_boundary.distance(polygon_in.boundary), min([p_boundary.distance(poly.boundary) for poly in polygon_out]) if len(polygon_out) > 0 else float('inf'))
                    delta_dist_node.append(min_dist-1)
            else:
                # outside geometry
                curr_min_dist = min(p.distance(polygon_in.boundary), min([p.distance(poly.boundary) for poly in polygon_out]) if len(polygon_out) > 0 else float('inf'))
                # check all sides
                delta_dist_node = []
                for d in direc:
                    p_boundary = Point(p.x + d[0] * delta_dist, p.y + d[1] * delta_dist)
                    min_dist = min(p_boundary.distance(polygon_in.boundary),
                                   min([p_boundary.distance(poly.boundary) for poly in polygon_out]) if len(
                                       polygon_out) > 0 else float('inf'))
                    if curr_min_dist > min_dist:
                        delta_dist_node.append(delta_dist)
                    else:
                        delta_dist_node.append(1e-9)
        else:
            delta_dist_node = [delta_dist for _ in range(4)]
        if Nd[i, 2] == 1:  # dx fixed
            lB_x = (1 - 1e-9) * Nd[i, 0]
            uB_x = (1 + 1e-9) * Nd[i, 0]
        else:
            lB_x = Nd[i, 0] - delta_dist_node[0]
            uB_x = Nd[i, 0] + delta_dist_node[1]
        if Nd[i, 3] == 1:
            lB_y = (1 - 1e-9) * Nd[i, 1]
            uB_y = (1 + 1e-9) * Nd[i, 1]
        else:
            lB_y = Nd[i, 1] - delta_dist_node[2]
            uB_y = Nd[i, 1] + delta_dist_node[3]
        lB_coord.extend([lB_x, lB_y])
        uB_coord.extend([uB_x, uB_y])
    lB_all = [1/scaling*10] + lB_coord + (-1*a_C*sC/scaling).tolist() + (-1*a_S_C*sC/scaling).tolist() #*max(1,lF_prev/1e2)
    uB_all = [float('inf')] + uB_coord + (0*a_C).tolist() + (a_S_T*sS/scaling).tolist()

    def equilibrium_function(X):
        # returns B*f = f
        # assemble Nd
        Nd_new = jax_np.array(X[1:Nd.shape[0] * 2 + 1]).reshape(Nd.shape[0], 2)
        # update Cn
        CnC_new = []
        for i in range(len(CnC)):
            n1, n2 = int(CnC[i, 0]), int(CnC[i, 1])
            dx, dy = Nd_new[n2, 0] - Nd_new[n1, 0], Nd_new[n2, 1] - Nd_new[n1, 1]
            CnC_new.append([n1, n2, jax_np.sqrt(dx ** 2 + dy ** 2), CnC[i, 3]])
        CnC_new = jax_np.array(CnC_new)
        CnS_new = []
        for i in range(len(CnS)):
            n1, n2 = int(CnS[i, 0]), int(CnS[i, 1])
            dx, dy = Nd_new[n2, 0] - Nd_new[n1, 0], Nd_new[n2, 1] - Nd_new[n1, 1]
            CnS_new.append([n1, n2, jax_np.sqrt(dx ** 2 + dy ** 2), CnS[i, 3], CnS[i, 4]])
        CnS_new = jax_np.array(CnS_new)
        # calculate equilibrium matrix B
        BC = calcB_jax(Nd_new, CnC_new, dof)
        BS = calcB_jax(Nd_new, CnS_new, dof)
        # calculate forces
        f_int_C = X[Nd.shape[0] * 2 + 1:Nd.shape[0] * 2 + 1 + len(CnC_new)]
        f_int_S = X[Nd.shape[0] * 2 + 1 + len(CnC_new):Nd.shape[0] * 2 + 1 + len(CnC_new) + len(CnS_new)]
        return (BC @ f_int_C + BS @ f_int_S - f * X[0]) * scaling  # *scaling # all is equally scaled by scaling

    def geometry_constraint(X):
        # check that members are inside polygon_in and outside polygon_out
        # assemble Nd
        eps = 1e-9
        Nd_new = jax_np.array(X[1:Nd.shape[0] * 2 + 1]).reshape(Nd.shape[0], 2)
        penalty = jax_np.array([0])
        for i in range(len(CnC)):
            n1, n2 = Nd_new[(CnC[i, 0]).astype(int)], Nd_new[(CnC[i, 1]).astype(int)]
            pts = []
            for j in range(polygon_in_jax.shape[0]):
                pts.append(intersection_point_jax(n1, n2, polygon_in_jax[j - 1], polygon_in_jax[j]))
            pts = jax_np.array(pts)
            sorted_pts = pts[jax_np.lexsort((-1*pts[:, 1], -1*pts[:, 0]))]
            dist = jax_np.sqrt(
                (sorted_pts[1][0] - sorted_pts[1][0]) ** 2 + (sorted_pts[2][1] - sorted_pts[1][1]) ** 2+eps)
            penalty += dist
            # if it is inside polygon_out = crosses boundary of inner polygons
            for poly in polygon_out_jax:
                pts = []
                for j in range(poly.shape[0]):
                    pts.append(intersection_point_jax(n1, n2, poly[j - 1], poly[j]))
                pts = jax_np.array(pts)
                sorted_pts = pts[jax_np.lexsort((-1 * pts[:, 1], -1 * pts[:, 0]))]
                dist = jax_np.sqrt(
                    (sorted_pts[0][0] - sorted_pts[1][0]) ** 2 + (sorted_pts[0][1] - sorted_pts[1][1]) ** 2+eps)
                penalty += dist
        return penalty
    def constraints(X):
        return jax_np.concatenate((equilibrium_function(X), geom_constraint_factor*geometry_constraint(X)))
    # Objective function
    optim_func = lambda X: -1*X[0]*scaling #load factor, + 0.01 * W @ np.abs(X[Nd.shape[0]*2+1:Nd.shape[0]*2+1+len(CnC)])
    # Initial guess
    x0 = jax_np.zeros(Nd.shape[0]*2 + 1 + len(CnC) + len(CnS))
    x0 = x0.at[0].set(1/scaling)  # Load factor
    x0 = x0.at[1:Nd.shape[0]*2+1].set(Nd[:,:2].flatten())  # Node coordinates
    x0 = x0.at[Nd.shape[0] * 2 + 1:Nd.shape[0] * 2 + 1 + len(CnC)].set(-1*jax_np.ones(len(CnC))/scaling)  # Internal forces in concrete struts
    x0 = x0.at[Nd.shape[0] * 2 + 1 + len(CnC):Nd.shape[0] * 2 + 1 + len(CnC) + len(CnS)].set(jax_np.ones(len(CnS))/scaling)  # Internal forces in steel ties
    # IPOPT requires gradients
    obj_jit = jax.jit(optim_func)
    obj_grad = jax.jit(jax.grad(obj_jit))
    obj_hess = jax.jit(jax.jacfwd(jax.jacrev(obj_jit)))
    con_jit = jax.jit(constraints)
    con_jac = jax.jit(jax.jacfwd(con_jit))
    con_hess = jax.jacfwd(jax.jacrev(con_jit))
    con_hess_vp = jax.jit(lambda x, v: con_hess(x) @ v)
    cons = [{'type': 'eq', 'fun': con_jit, 'jac': con_jac, 'hess': con_hess_vp}]
    bnds = [(lB_all[i], uB_all[i]) for i in range(len(lB_all))]
    result = cyipopt.minimize_ipopt(obj_jit, jac = obj_grad, hess = obj_hess, x0 = x0, bounds = bnds, constraints = cons, tol = 1e-8, options={'disp': 1, 'check_derivatives_for_naninf': 'yes', 'nlp_scaling_method': 'gradient-based','hessian_approximation': 'limited-memory', 'maxiter': 30000}) # 'disp': 5 = shows all details
    if not result.success:
        print("Optimisation failed:", result.message)
    # Extract optimised values
    optimised_values = result.x
    print('Constraint error:', max(con_jit(optimised_values)), min(con_jit(optimised_values)))
    print('Equilibrium error with func:', max(equilibrium_function(optimised_values)), min(equilibrium_function(optimised_values)))
    #print('Is it nonconvex?', jax.scipy.linalg.eigh(con_hess(optimised_values), eigvals_only=True))
    print('Geometry error with func:', geometry_constraint(optimised_values)[0])
    load_factor = optimised_values[0]*scaling
    print('Final load factor:', load_factor)
    Nd_optimised = optimised_values[1:Nd.shape[0]*2+1].reshape(Nd.shape[0], 2)
    # update CnC and CnS
    CnC_optimised = []
    for i in range(len(CnC)):
        n1, n2 = int(CnC[i, 0]), int(CnC[i, 1])
        dx, dy = Nd_optimised[n2, 0] - Nd_optimised[n1, 0], Nd_optimised[n2, 1] - Nd_optimised[n1, 1]
        CnC_optimised.append([n1, n2, np.sqrt(dx ** 2 + dy ** 2), CnC[i, 3]])
    CnC_optimised = np.array(CnC_optimised)
    CnS_optimised = []
    for i in range(len(CnS)):
        n1, n2 = int(CnS[i, 0]), int(CnS[i, 1])
        dx, dy = Nd_optimised[n2, 0] - Nd_optimised[n1, 0], Nd_optimised[n2, 1] - Nd_optimised[n1, 1]
        CnS_optimised.append([n1, n2, np.sqrt(dx ** 2 + dy ** 2), CnS[i, 3], CnS[i, 4]])
    CnS_optimised = np.array(CnS_optimised)
    f_int_C_optimised = optimised_values[Nd.shape[0]*2+1:Nd.shape[0]*2+1+len(CnC)]*scaling
    f_int_S_optimised = optimised_values[Nd.shape[0]*2+1+len(CnC):Nd.shape[0]*2+1+len(CnC)+len(CnS)]*scaling
    BC = calcB(Nd_optimised, CnC_optimised, dof)  # len(f) x len(CnC) = number of struts
    BS = calcB(Nd_optimised, CnS_optimised, dof)  # len(f) x len(CnS) = number of ties
    print('Check equilibrium (should be close to 0):',
          np.sum(np.abs(BC @ f_int_C_optimised + BS @ f_int_S_optimised - f * load_factor)))
    return Nd_optimised, CnC_optimised, CnS_optimised, f_int_C_optimised, f_int_S_optimised, load_factor