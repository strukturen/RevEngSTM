# Reverse Engineering Strut-and-Tie Models
# ================================
# Karin Yu, 2026
# Base file to be coupled with example parameters

__author__ = 'Karin Yu'

# load custom source files
from src.problem import *
from src.plotter import *
from src.solver import *
from src.geom_optim import *
import src.stm_trusssystem as TS
import sys # for command line arguments
import time
import os

if len(sys.argv) > 1:
    print('Command arguments passed')
    example_name_file = sys.argv[1] # name of the example file
    eps_init = float(sys.argv[2]) # for clustering
    if len(sys.argv) > 3:
        add_args = sys.argv[3]
    else:
        add_args = ""
else:
    example_name_file = 'Example_deep_beam_Schlaich_exp.py'
    eps_init = 0.5  # for clustering; DEB: 0.3, DEB with opening: 0.3, Deep Beam Schlaich: 0.5, Wall 2 openings: 0.5
    add_args = "C"

# Modify this information
# OPTIM: For optimisation loop
# PLOT: For plotting
mode = 'OPTIM'
# save file name
file_name = '_RevEngSTM.npz'
iter_geom_optim = 10
mp_ls = [100] # List of member penalties
figsize_factor = 3.0
loadfactor_accept = 0.5

# extract parameters:
# - geometry
# - loads
# - boundary conditions
# - mesh.py sizes
# - material properties
# - hyperparameters: thresholds and number of iterations
exec(open(example_name_file).read())

if mode == 'OPTIM':
    print('##### Assemble problem'+'data/scaling/' + example_name + file_name[:-4]+'_GS.npz')
    if not os.path.exists('data/' + example_name + file_name[:-4]+'_GS.npz'):
        bc_coords = np.concatenate((dof_coords, force_coords), axis=0)
        # assemble problem
        points = get_points(polygon_in, polygon_out, mesh_coarse, mesh_fine, add_x = add_x, add_y = add_y)
        print('# Points created')
        Nd = get_nodes(points, polygon_in)
        print('# Nodes created: %d nodes' %(len(Nd)))
        dof, f = set_boundary_conditions(Nd, dof_coords, dofs, force_coords, forces)
        print('# Boundary conditions and loads set', len(dof)-sum(dof), max(f), min(f))
        time_start = time.time()
        CnC, CnS = set_reinforcement_layout(Nd, reinf_list, max_incl, polygon_in, polygon_out, ac_min, as_min)
        time_end = time.time()
        np.savez('data/' + example_name + file_name[:-4]+'_GS.npz', bc_coords = bc_coords, Nd = Nd, dof = dof, f = f, CnC = CnC, CnS = CnS)
        print('# Reinforcement layout created:', time_end - time_start,' s')
    else:
        print('# Load existing ground structure')
        data_GS = np.load('data/' + example_name + file_name[:-4]+'_GS.npz')
        bc_coords = data_GS['bc_coords']
        Nd = data_GS['Nd']
        dof = data_GS['dof']
        f = data_GS['f']
        CnC = data_GS['CnC']
        CnS = data_GS['CnS']
    print('# Number of struts: %d, and ties: %d' % (len(CnC), len(CnS)))
    qS = np.zeros((len(CnS)))
    qC = np.zeros((len(CnC)))
    plotTruss(Nd, CnC, CnS, qC, qS, geom, polygon_out = geom_out, plt_only_sign = True, size_factor = figsize_factor*mesh_coarse, savefig = 'results/'+example_name+'_GS.pdf') #
    #plt.show()
    print('##### Part 1: Layout optimisation')
    # solve convex optimization problem
    # objective: f_int_S**2@fact_S-100*load_factor+0.2*cvx.norm(f_int_C,1)
    for mp in mp_ls:
        print('### Member penalty: %d' % mp)
        CnC_1, CnS_1, qC, qS, lF, _ = solveProblem(Nd, CnC, CnS, f, dof, fsd, fcd, bc_coords, reinforcement = reinf_list, max_incl_prev = max_incl, polygon_in = polygon_in, polygon_out = polygon_out, ac_min = ac_min, max_num_iter=max_num_iter, max_threshold_S=max_threshold_S, max_threshold_C=max_threshold_C, member_penalty = mp)
        checkEquilibrium(Nd, CnC_1, CnS_1, lF*f, dof, qC, qS)
    plotTruss(Nd, CnC_1, CnS_1, qC, qS, polygon_in=geom, polygon_out=geom_out, size_factor=figsize_factor*mesh_coarse, plt_only_sign=False,loadFactor = lF, savefig='results/'+example_name+'_result_S1.pdf') #
    plt.title('After CVX, LF = {:.0f} kN'.format(lF[0]/1e3))
    CnC_prev = CnC_1.copy()
    CnS_prev = CnS_1.copy()
    qC_prev = qC.copy()
    qS_prev = qS.copy()
    lF_prev = lF[0]
    Nd_prev = Nd.copy()
    CnC_cvx = CnC_1.copy()
    CnS_cvx = CnS_1.copy()
    qC_cvx = qC.copy()
    qS_cvx = qS.copy()
    lF_cvx = lF[0]
    Nd_cvx = Nd.copy()
    CnC_best = CnC_1.copy()
    CnS_best = CnS_1.copy()
    qC_best = qC.copy()
    qS_best = qS.copy()
    lF_best = lF[0]
    Nd_best = Nd.copy()
    num_bc = len(dofs)*len(dofs[0])-sum([sum(d) for d in dofs])+len(forces)
    cluster_bool = True
    # X number of geometry optimization and clustering iterations
    for j in range(iter_geom_optim):
        print('################################################################################')
        print('##### Part 2: Node clustering and geometry optimization iteration %d' % (j+1))
        print('# Number of struts: %d, and ties: %d' % (len(CnC_prev), len(CnS_prev)))
        # 1. remove redundant members which have the same force and connecting node
        CnC_temp, CnS_temp, _, _ = reduce_members(Nd_prev, CnC_prev, CnS_prev, qC_prev, qS_prev, max_threshold_C, max_threshold_S)#*(iter_geom_optim-j)/iter_geom_optim
        # 2. add nodes at intersection points of intersecting members
        if j == 0:
            Nd_prev, CnC_temp, CnS_temp, _ = intersecting_members(Nd_prev, CnC_temp, CnS_temp) # adds nodes but does not remove them
        # 3. cluster nodes
        print('# Before clustering: Number of nodes: %d, struts: %d, ties: %d' % (Nd_prev.shape[0], len(CnC_temp), len(CnS_temp)))
        Nd_clus, CnC_clus, CnS_clus = clusterNodes(Nd_prev, CnC_temp, CnS_temp, bc_coords,  eps = (eps_init+j/iter_geom_optim*0.5)*mesh_fine) #(0.3+j/iter_geom_optim*0.5)*mesh_fine
        print('# Number of clustered nodes: %d, struts: %d, ties: %d' % (Nd_clus.shape[0], len(CnC_clus), len(CnS_clus)))
        if cluster_bool: # another intersecting of members + clustering, can also be a variable
            numCross_Touch = 1
            k = 0
            while numCross_Touch != 0: # can also be a variable
                if k == 0:
                    Nd_clus, CnC_clus, CnS_clus, numCross_Touch = intersecting_members(Nd_clus[:,:2], CnC_clus, CnS_clus, check_touching = True, dist_tol = 0.15*mesh_fine)
                else:
                    Nd_clus, CnC_clus, CnS_clus, numCross_Touch = intersecting_members(Nd_clus[:, :2], CnC_clus, CnS_clus)
                print('# After %d. iteration adding intersection nodes; nodes: %d, number of struts: %d, and ties: %d' % (k+1, Nd_clus.shape[0], len(CnC_clus), len(CnS_clus)))
                Nd_clus, CnC_clus, CnS_clus = clusterNodes(Nd_clus, CnC_clus, CnS_clus, bc_coords,  eps = (eps_init+j/iter_geom_optim*0.5)*mesh_fine)
                print('# Number of %d. time clustered nodes: %d, struts: %d, ties: %d' % (k+1, Nd_clus.shape[0], len(CnC_clus), len(CnS_clus)))
                k += 1
            if k == 1 and numCross_Touch == 0:
                cluster_bool = False
        # 4. optimise geometry
        dof_red, f_red = set_boundary_conditions(Nd_clus, dof_coords, dofs, force_coords, forces)
        Nd_new, CnC_new, CnS_new, qC_new, qS_new, lF_new = optimiseGeometry(Nd_clus, CnC_clus, CnS_clus, f_red, dof_red, fcd, fsd, polygon_in=polygon_in, polygon_out=polygon_out,delta_dist = 0.25 * (iter_geom_optim-j+1)/iter_geom_optim*cnom+1e-3, scaling =10000) #reinforcement = reinf_list, max_incl_prev = max_incl, polygon_in = polygon_in, polygon_out = polygon_out, ac_min = ac_min, max_num_iter=max_num_iter, max_threshold_S=max_threshold_S, max_threshold_C=max_threshold_C)
        print('# After geometry optimization: Number of struts: %d, and ties: %d' % (len(CnC_new), len(CnS_new)))
        # update for next iteration
        CnC_prev = CnC_new.copy()
        CnS_prev = CnS_new.copy()
        qC_prev = qC_new.copy()
        qS_prev = qS_new.copy()
        lF_prev = lF_new.copy()
        Nd_prev = Nd_new.copy()
        if lF_new >= loadfactor_accept*lF_best:
            # 1. remove redundant members which have the same force and connecting node
            CnC_best = CnC_new.copy()
            CnS_best = CnS_new.copy()
            qC_best = qC_new.copy()
            qS_best = qS_new.copy()
            lF_best = lF_new.copy()
            Nd_best = Nd_new.copy()
    print('##### Final result after geometry optimisation and node clustering')
    print('# Number of nodes: %d, struts: %d, and ties: %d' % (Nd_best.shape[0], len(CnC_best), len(CnS_best)))
    # fig = plotTruss(Nd_best, CnC_best, CnS_best, qC_best, qS_best, polygon_in=polygon_in,
    #                polygon_out=polygon_out, size_factor=figsize_factor * mesh_coarse, plt_only_sign=True, savefig='results/'+example_name+'_result_GO_connectivity.pdf')
    # for i in range(Nd_best.shape[0]):
    #    plt.plot(Nd_best[i, 0], Nd_best[i, 1], '.', color='tab:orange', markersize=2)
    # plt.title('Final Connectivity')
    # Save data
    np.savez('data/' + example_name + file_name, Nd=Nd_best, CnC=CnC_best, CnS=CnS_best, qC=qC_best, qS=qS_best, lF=lF_best, Nd_cvx=Nd_cvx, CnC_cvx=CnC_cvx, CnS_cvx=CnS_cvx, qC_cvx=qC_cvx, qS_cvx=qS_cvx, lF_cvx=lF_cvx)
    dof_red, f_red = set_boundary_conditions(Nd_best, dof_coords, dofs, force_coords, forces)
    checkEquilibrium(Nd_best, CnC_best, CnS_best, lF_best*f_red, dof_red, qC_best, qS_best)
    fig = plotTruss(Nd_best, CnC_best, CnS_best, qC_best, qS_best, polygon_in=geom,
                    polygon_out=geom_out, size_factor=figsize_factor * mesh_coarse, plt_only_sign=False, loadFactor=lF_best, savefig='results/'+example_name+'_result_S2.pdf')
    for i in range(Nd_best.shape[0]):
        plt.plot(Nd_best[i, 0], Nd_best[i, 1], '.', color='tab:orange', markersize=2)
    plt.title('Final STM LF = {:.0f} kN'.format(lF_best/1e3))
    #plt.show()
elif mode == 'PLOT':
    data = np.load('data/' + example_name + file_name)
    Nd = data['Nd']
    CnC = data['CnC']
    CnS = data['CnS']
    qC = data['qC']
    qS = data['qS']
    lF = data['lF']
    dof_red, f_red = set_boundary_conditions(Nd, dof_coords, dofs, force_coords, forces)
    print('# Data loaded from file')
    print('# Material properties\n# fcd = %f MPa, fsd = %f MPa' % (fcd, fsd))
    checkEquilibrium(Nd, CnC, CnS, lF * f_red, dof_red, qC, qS)
    # connectivity
    # plotTruss(Nd, CnC, CnS, qC, qS, polygon_in=polygon_in, polygon_out=polygon_out, size_factor=figsize_factor * mesh_coarse, plt_only_sign=True, loadFactor=lF)
    # plt.show()
    # STM result
    plotTruss(Nd, CnC, CnS, qC, qS, polygon_in=geom, polygon_out=geom_out, size_factor=figsize_factor * mesh_coarse, plt_only_sign=False, loadFactor=lF)
    plt.show()