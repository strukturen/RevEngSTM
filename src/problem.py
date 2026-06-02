# Reverse Engineering Strut-and-Tie Models
# ================================
# Karin Yu, 2026

# Module: Separate struts and ties
# Problem

__author__ = 'Karin Yu'

import numpy as np
from shapely.geometry import Point, LineString, Polygon
from rtree.index import Index
import itertools
from src.utils import *

def create_bounds(polygon_in, polygon_out, mesh_coarse, mesh_fine = 50, add_x = [], add_y = [], rnd_digit = 2):
    # creates boundaries for ground structure
    # polygon_in is only one polygon
    # polygon_out is a list of polygons
    x_bounds = set()
    y_bounds = set()
    for pts in polygon_in.exterior.coords:
        x_bounds.add(pts[0])
        y_bounds.add(pts[1])
    for poly in polygon_out:
        for pts in poly.exterior.coords:
            x_bounds.add(pts[0])
            y_bounds.add(pts[1])
    x_bounds = sorted(list(x_bounds)+add_x)
    y_bounds = sorted(list(y_bounds)+add_y)
    mesh_coarse_factor = int(mesh_coarse / mesh_fine)
    # reduce node density
    def get_bounds(bounds, mesh_coarse_factor):
        # gets additional bounds if mesh.py does not fit into the distance between two boundary points
        i = 1
        while i < len(bounds):
            if (bounds[i] - bounds[i - 1]) % mesh_coarse != 0:
                factor = 1
                while ((bounds[i] - bounds[i - 1]) % mesh_coarse) * factor % mesh_fine != 0 and (
                        bounds[i] - ((bounds[i] - bounds[i - 1]) % mesh_coarse) * factor) > bounds[i - 1]:
                    factor += 1
                    if factor >= mesh_coarse_factor:
                        break
                bounds[i:] = [bounds[i] - ((bounds[i] - bounds[i - 1]) % mesh_coarse) * factor] + bounds[i:]
                i += 1
            i += 1
        return bounds
    x_bounds = get_bounds(x_bounds, mesh_coarse_factor)
    y_bounds = get_bounds(y_bounds, mesh_coarse_factor)
    x_bounds = [round(e, rnd_digit) for e in x_bounds]
    y_bounds = [round(e, rnd_digit) for e in y_bounds]
    return sorted(list(set(x_bounds))), sorted(list(set(y_bounds)))

def get_points(polygon_in, polygon_out, mesh_coarse, mesh_fine = 50, add_x = [], add_y = []):
    # creates points for ground structure
    # polygon_in is only one polygon
    # polygon_out is a list of polygons
    # returns points in geometrical continuum
    x_bounds, y_bounds = create_bounds(polygon_in, polygon_out, mesh_coarse, mesh_fine = mesh_fine, add_x = add_x, add_y = add_y)
    print('X bounds:', x_bounds)
    print('Y bounds:', y_bounds)
    def get_ticks(bounds):
        tol = 1e-6  # for float comparison
        ticks = []

        for i in range(1, len(bounds)):
            interval = bounds[i] - bounds[i - 1]

            # Find best fitting mesh size
            step = mesh_coarse
            factor = 0
            while (interval % step) > tol:
                factor += 1
                candidate = mesh_coarse - factor * mesh_fine
                if candidate <= mesh_fine:
                    # No rounded divisor found — just use mesh_fine
                    step = mesh_fine
                    break
                step = candidate

            segment = np.arange(bounds[i - 1], bounds[i], step)
            ticks.append(segment)
            ticks.append([bounds[i]])  # ✅ always close each interval

        ticks = np.concatenate(ticks)
        ticks = np.unique(np.round(ticks, 6))  # remove duplicates at shared bounds
        return ticks
    x_ticks = get_ticks(x_bounds)
    y_ticks = get_ticks(y_bounds)
    # create points
    pts = []
    for i in range(x_ticks.shape[0]):
        for j in range(y_ticks.shape[0]):
            x, y = x_ticks[i], y_ticks[j]
            if all([not poly.contains(Point(x, y)) for poly in polygon_out]):
                pts.append(Point(x, y))
    return pts

def get_nodes(pts, polygon_in):
    return np.array([[pt.x, pt.y] for pt in pts if polygon_in.intersects(pt)])

def set_boundary_conditions(Nd,dof_coords, dofs, force_coords, forces):
    # sets boundary conditions
    # returns dof and f
    dof = np.ones((len(Nd),2)) # list of DOFs, 0 = fixed, 1 = free
    f = [] # list of external forces
    # set up dictionary
    dof_dict = {}
    for i in range(dof_coords.shape[0]):
        # round coordinate points to avoid numerical issues
        dof_dict[(round(dof_coords[i,0], 2), round(dof_coords[i,1], 2))] = dofs[i]
    for i, nd in enumerate(Nd):
        if (round(nd[0],2), round(nd[1],2)) in dof_dict:
            dof[i,:] = dof_dict[(round(nd[0],2), round(nd[1],2))]
        force_assigned = False
        for fi in range(force_coords.shape[0]):
            if (round(nd[0],2), round(nd[1],2)) == (round(float(force_coords[fi,0]),2), round(float(force_coords[fi,1]),2)):
                f += [forces[fi][0], forces[fi][1]]
                force_assigned = True
                break
        if not force_assigned: f += [0,0]
    return np.array(dof).flatten(), np.array(f).flatten()

class DynamicRTree:
    def __init__(self):
        self.idx = Index()
        self.segments = {}
        self.next_id = 0

    def insert(self, seg):
        i = self.next_id
        self.next_id += 1
        self.segments[i] = seg
        self.idx.insert(i, seg.bounds)
        return i

    def delete(self, i):
        seg = self.segments.pop(i, None)
        if seg is not None:
            self.idx.delete(i, seg.bounds)

    def has_overlap(self, new_seg, tol=1e-3):
        for i in self.idx.intersection(new_seg.bounds):
            seg = self.segments[i]
            # use overlaps to exclude endpoint touches; add length threshold if needed
            if seg.intersects(new_seg) and seg.intersection(new_seg).length > tol:
                return True
        return False

def set_reinforcement_layout(Nd, reinforcement, max_incl, polygon_in, polygon_out, ac_min, min_reinf = 0, tol = 1., dist_tol = 1):
    # separates ties from struts in the ground structure
    # ties are defined based on the reinforcement layout (of concentrated reinforcement)
    # struts are defined based on the allowed maximum angle
    # pre-process all nodes on reinforcement
    reinf_list, reinf_segments = preprocess_reinf(Nd, reinforcement, tol)
    # CnC structure: [Node index i, Node index j, length, concrete area for compression]
    # CnS structure: [Node index i, Node index j, length, concrete area for compression, steel area for tension]
    CnC, CnS = [], []
    tie_segments = []
    # generate CnS
    for i, j in itertools.combinations(range(len(Nd)), 2): # unique combinations
        # reinforcement
        dx, dy = abs(Nd[i, 0] - Nd[j, 0]), abs(Nd[i, 1] - Nd[j, 1])
        if any([x and y for x,y in zip(reinf_list[i], reinf_list[j])]):
            seg = LineString([Nd[i], Nd[j]])
            if any([seg.distance(Point(Nd[k])) <= dist_tol for k in range(Nd.shape[0]) if (not all(Nd[k] == Nd[i]) and not all(Nd[k] == Nd[j]))]): continue
            # get indices of corresponding reinforcement
            true_indices = [k for k, (x, y) in enumerate(zip(reinf_list[i], reinf_list[j])) if x and y]
            # check if the member does not cross a node
            CnS.append([i,j,np.sqrt(dx**2+dy**2), ac_min, reinforcement[true_indices[0], 4] + min_reinf]) # 4 bars per tie
            tie_segments.append(LineString([Nd[i], Nd[j]]))
    all_nodes = [Point(Nd[i]) for i in range(len(Nd))]
    # generate CnC
    for i, j in itertools.combinations(range(len(Nd)), 2):  # unique combinations
        dx, dy = abs(Nd[i, 0] - Nd[j, 0]), abs(Nd[i, 1] - Nd[j, 1])
        # concrete
        if any([x and y for x, y in zip(reinf_list[i], reinf_list[j])]):
            # reinforcement
            continue
        elif (dx != 0 or dy != 0) and ((dx == dy) or ((dx == 0 or dy == 0) or (dx/dy <= max_incl) and (dy/dx <= max_incl))):
            seg = LineString([Nd[i], Nd[j]])
            if (polygon_in.contains(seg) or polygon_in.boundary.contains(seg)) and all(
                    [not LineInPolygon(seg, poly) for poly in polygon_out]) and not any([seg.intersects(p) for p in all_nodes if not p in [Point(seg.coords[0]),Point(seg.coords[-1])]]):
                CnC.append([i, j, np.sqrt(dx ** 2 + dy ** 2), ac_min])
    CnC, CnS = np.array(CnC), np.array(CnS)
    return CnC, CnS

def adapt_layout(Nd, Nd_active, CnS, CnC, qS, bc_coords, reinforcement, max_incl_prev, max_incl_new , polygon_in, polygon_out, ac_min, tol = 1., threshold_S = 1, min_strut_angle = 25/180*np.pi):
    # adapts the layout due to increased inclination and strut angle requirement
    # separates ties from struts in the ground structure
    # ties are defined based on the reinforcement layout (of concentrated reinforcement)
    # struts are defined based on the allowed maximum angle
    # pre-process all nodes on reinforcement
    reinf_list, reinf_segments = preprocess_reinf(Nd, reinforcement, tol)
    tie_list = [[False] * len(CnS) for _ in range(len(Nd))]
    tie_segments = [LineString(Nd[CnS[i, [0, 1]].astype(int), :]) for i in range(len(CnS)) if qS[i] >= threshold_S]
    for i in range(len(Nd)):
        for r in range(len(tie_segments)):
            if tie_segments[r].distance(Point(Nd[i])) < tol:
                tie_list[i][r] = True
    # determine new struts due to increased inclination
    DynamicRTree_C = DynamicRTree()
    CnC_new = []
    for i, j in itertools.combinations(range(len(Nd)), 2): # unique combinations
        if not Nd_active[i] or not Nd_active[j]: continue
        dx, dy = abs(Nd[i, 0] - Nd[j, 0]), abs(Nd[i, 1] - Nd[j, 1])
        # reinforcement
        if any([x and y for x,y in zip(reinf_list[i], reinf_list[j])]):
            continue
        # concrete
        elif (dx != 0 and dy != 0) and (((max_incl_prev < dx/dy <= max_incl_new) and (dy/dx <= max_incl_new)) or ((max_incl_prev < dy/dx <= max_incl_new) and (dx/dy <= max_incl_new))): #
            seg = LineString([Nd[i], Nd[j]])
            # check if it fulfills strut angle constraint
            if not strut_constraint(tie_segments, tie_list, [i, j], seg, min_strut_angle, bc_coords):
                if (polygon_in.contains(seg) or polygon_in.boundary.contains(seg)) and all(
                        [not LineInPolygon(seg, poly) for poly in polygon_out]) and not DynamicRTree_C.has_overlap(seg):#not any([seg.intersects(p) for p in all_nodes if not p in [Point(seg.coords[0]),Point(seg.coords[-1])]]):#
                    CnC_new.append([i, j, np.sqrt(dx ** 2 + dy ** 2), ac_min])
                    DynamicRTree_C.insert(seg)
    print('Additional struts due to inclination:', len(CnC_new))
    CnC_new = np.array(CnC_new)
    return CnC_new

def reduce_members(Nd, CnC, CnS, qC, qS, max_threshold_C, max_threshold_S):
    # reduces members if they are the only two members at one node and have similar forces
    if CnC.shape[0] != qC.shape[0] or CnS.shape[0] != qS.shape[0]:
        raise ValueError('Number of members and forces do not match!')
    Nd_ls_C = [[] for _ in range(len(Nd))]
    Nd_ls_S = [[] for _ in range(len(Nd))]
    max_qC = max_threshold_C*min(qC)
    min_qS = max_threshold_S*max(qS)
    print('max C', max_qC, 'min S', min_qS)
    # apply thresholding first
    CnC = CnC[np.array(qC).flatten() <= max_qC]
    qC = qC[np.array(qC).flatten() <= max_qC]
    CnS = CnS[np.array(qS).flatten() >= min_qS]
    qS = qS[np.array(qS).flatten() >= min_qS]
    for i in range(len(CnC)):
        if qC[i] <= max_qC:
            Nd_ls_C[int(CnC[i, 0])].append(i)
            Nd_ls_C[int(CnC[i, 1])].append(i)
    for i in range(len(CnS)):
        if qS[i] >= min_qS:
            Nd_ls_S[int(CnS[i, 0])].append(i)
            Nd_ls_S[int(CnS[i, 1])].append(i)
    # reduce members
    redundant_pairs_C = []
    redundant_pairs_S = []
    for i in range(len(Nd)):
        # only account for cases of two members at one node
        if len(Nd_ls_C[i]) == 2:
            c0, c1 = Nd_ls_C[i][0], Nd_ls_C[i][1]
            C_angle0 = getAngle(Nd[int(CnC[c0,0]), 0] - Nd[int(CnC[c0,1]), 0], Nd[int(CnC[c0,0]), 1] - Nd[int(CnC[c0,1]), 1])  # arctan(dy/dx)
            C_angle0 = boundAngle(C_angle0, factor_max_angle = 1)
            C_angle1 = getAngle(Nd[int(CnC[c1, 0]), 0] - Nd[int(CnC[c1, 1]), 0],
                                Nd[int(CnC[c1, 0]), 1] - Nd[int(CnC[c1, 1]), 1])  # arctan(dy/dx)
            C_angle1 = boundAngle(C_angle1, factor_max_angle = 1)
            if abs(qC[c0]-qC[c1]) <= abs(max_qC)*0.1 and abs(C_angle0-C_angle1) <= 1/180*np.pi:
                redundant_pairs_C.append([c0,c1])
        if len(Nd_ls_S[i]) == 2:
            c0, c1 = Nd_ls_S[i][0], Nd_ls_S[i][1]
            S_angle0 = getAngle(Nd[int(CnS[c0, 0]), 0] - Nd[int(CnS[c0, 1]), 0],
                                Nd[int(CnS[c0, 0]), 1] - Nd[int(CnS[c0, 1]), 1])  # arctan(dy/dx)
            S_angle0 = boundAngle(S_angle0, factor_max_angle = 1)
            S_angle1 = getAngle(Nd[int(CnS[c1, 0]), 0] - Nd[int(CnS[c1, 1]), 0],
                                Nd[int(CnS[c1, 0]), 1] - Nd[int(CnS[c1, 1]), 1])  # arctan(dy/dx)
            S_angle1 = boundAngle(S_angle1, factor_max_angle = 1)
            if abs(qS[c0] - qS[c1]) <= abs(min_qS) * 0.1 and abs(S_angle0 - S_angle1) <= 1 / 180 * np.pi:
                redundant_pairs_S.append([c0,c1])
    redundant_pairs_C = np.array(redundant_pairs_C, dtype= int)
    redundant_pairs_S = np.array(redundant_pairs_S, dtype=int)
    print('Number of redundant struts:', redundant_pairs_C.shape[0])
    print('Number of redundant ties:', redundant_pairs_S.shape[0])
    # remove redundant pairs
    def remove_redundant_pairs(redundant_pairs, Cn):
        removed_members = []
        for i in range(redundant_pairs.shape[0]):
            pair = [int(redundant_pairs[i][0]), int(redundant_pairs[i][1])]
            # find common node
            common_node = Cn[pair[0], 0] if (Cn[pair[0], 0] == Cn[pair[1],0] or Cn[pair[0], 0] == Cn[pair[1],1]) else Cn[pair[0], 1]
            # remove pair[1] and modify pair[0]
            if Cn[pair[0], 0] == common_node:
                idx_to_replace = 0
            else:
                idx_to_replace = 1
            # replace index
            if common_node == Cn[pair[1],0]:
                Cn[pair[0], idx_to_replace] = Cn[pair[1],1]
            else:
                Cn[pair[0], idx_to_replace] = Cn[pair[1],0]
            # remove pair[1]
            removed_members.append(pair[1])
            redundant_pairs[i:][redundant_pairs[i:] == pair[1]] = pair[0]
        return removed_members
    members_to_remove_C = remove_redundant_pairs(redundant_pairs_C, CnC)
    members_to_remove_S = remove_redundant_pairs(redundant_pairs_S, CnS)
    print('Number of removed redundant struts:', len(members_to_remove_C))
    print('Number of removed redundant ties:', len(members_to_remove_S))
    CnC = np.delete(CnC, members_to_remove_C, axis=0)
    qC = np.delete(qC, members_to_remove_C, axis=0)
    CnS = np.delete(CnS, members_to_remove_S, axis=0)
    qS = np.delete(qS, members_to_remove_S, axis=0)
    return CnC, CnS, qC, qS

def strut_constraint(tie_segments, tie_list, strut_indices, strut_segment, min_strut_angle, bc_coords):
    # returns boolean if True or False
    # if strut angle is smaller than 25 degrees = True = Constraint violated
    ni, nj = strut_indices
    # check if they have common nodes
    common_nodes = []
    if any(tie_list[ni]):
        common_nodes.append(ni)
        common_node_coord = strut_segment.coords[0]
        C_angle = getAngleofLineString(strut_segment, 0)
    elif any(tie_list[nj]):
        common_nodes.append(nj)
        common_node_coord = strut_segment.coords[1]
        C_angle = getAngleofLineString(strut_segment, 1)
    else:
        return False
    for common_node in common_nodes:
        reinf_indices = [k for k, x in enumerate(tie_list[common_node]) if x]
        C_angle = boundAngle(C_angle, factor_max_angle=2)
        for r in reinf_indices:
            if common_node_coord == tie_segments[r].coords[0]:
                reinf_angle = getAngleofLineString(tie_segments[r], 0)
            else:
                reinf_angle = getAngleofLineString(tie_segments[r], 1)
            reinf_angle = boundAngle(reinf_angle, factor_max_angle=2)
            # check if nodes are part of boundary conditions
            if common_node_coord in bc_coords:
                return False
            if abs(C_angle - reinf_angle) <= min_strut_angle or 2*np.pi-abs(C_angle - reinf_angle) <= min_strut_angle:
                return True
    return False

def getAngleofLineString(line: LineString, first_coord = 0):
    if first_coord == 0:
        return getAngle(line.coords[1][0] - line.coords[0][0],
                           line.coords[1][1] - line.coords[0][1])
    elif first_coord == 1:
        return getAngle(line.coords[0][0] - line.coords[1][0],
                           line.coords[0][1] - line.coords[1][1])

def preprocess_reinf(Nd, reinforcement, tol = 1. ):
    # pre-process all nodes on reinforcement to obtain that a node has a reinforcement (reinf_list)
    # and to save the LineStrings of the reinforcement segments
    reinf_list = [[False] * len(reinforcement) for _ in range(len(Nd))]
    reinf_segments = [LineString([(round(float(reinforcement[i, 0]), 2), round(float(reinforcement[i, 1]), 2)),
                                  (round(float(reinforcement[i, 2]), 2), round(float(reinforcement[i, 3]), 2))]) for i in range(reinforcement.shape[0])]
    for i in range(len(Nd)):
        for r in range(len(reinf_segments)):
            if reinf_segments[r].distance(Point(Nd[i])) < tol:
                reinf_list[i][r] = True
    return reinf_list, reinf_segments
