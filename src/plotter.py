# Reverse Engineering Strut-and-Tie Models
# ================================
# Karin Yu, 2026

# Plotter

__author__ = 'Karin Yu'

import numpy as np
import matplotlib.pyplot as plt
import src.stm_trusssystem as TS

def plotTruss(Nd, CnC, CnS, qC, qS, polygon_in, polygon_out=None, size_factor = 1, savefig = None, fig_size = None, force_plot_factor = 3e5, plt_only_sign = False, incl_nodes = False, loadFactor = None):
    # Plots strut-and-tie model
    # if plt_only_sign is True, only plot sign of forces, so it shows the connectivity
    # else it shows the force magnitudes as line thickness
    # force_plot_factor = 3e5 --> thickness of figures
    if polygon_out is None:
        polygon_out = []
    if fig_size is None:
        coord_arr = np.array(polygon_in.exterior.coords)
        xmin, xmax, ymin, ymax = np.min(coord_arr[:,0]), np.max(coord_arr[:,0]), np.min(coord_arr[:,1]), np.max(coord_arr[:,1])
        fig_size = ((xmax-xmin)/size_factor,(ymax-ymin)/size_factor)
    fig = plt.figure(figsize=fig_size)
    for i in [i for i in range(qC.shape[0])]:
        pos = Nd[CnC[i, [0,1]].astype(int),:]
        if plt_only_sign:
            plt.plot(pos[:,0], pos[:,1], linestyle = '-', color='tab:green',  linewidth = 0.2)
        else:
            plt.plot(pos[:,0], pos[:,1], color='tab:green', linewidth = abs(qC[i].item()/force_plot_factor))
        if incl_nodes:
            plt.plot(pos[0,0], pos[0,1],'.', color = 'k', alpha = 0.2)
            plt.plot(pos[1,0], pos[1,1],'.', color = 'k', alpha = 0.2)
    for i in [i for i in range(qS.shape[0])]:
        pos = Nd[CnS[i, [0,1]].astype(int),:]
        if plt_only_sign:
            if qS[i] < 0:
                plt.plot(pos[:,0], pos[:,1], linestyle = '-', color='tab:green', linewidth = 0.2) #marker = '.',
            elif qS[i] >= 0:
                plt.plot(pos[:, 0], pos[:, 1], color='tab:blue', linewidth= 1)
        else:
            if qS[i] < 0:
                plt.plot(pos[:,0], pos[:,1], color='tab:green', linewidth = abs(qS[i].item()/force_plot_factor))
            elif qS[i] > 0:
                plt.plot(pos[:, 0], pos[:, 1], color='tab:blue', linewidth=abs(qS[i].item() / force_plot_factor))
                #plt.text(sum(pos[:, 0]) / 2, sum(pos[:, 1]) / 2, str(int(qS[i]/1e3))+"kN", fontsize=8)  # marker = '.',str(int(CnS[i,4])) + "mm2"
        if incl_nodes:
            plt.plot(pos[0,0], pos[0,1],'.', color = 'k', alpha = 0.2)
            plt.plot(pos[1,0], pos[1,1],'.', color = 'k', alpha = 0.2)
    x_list, y_list = [polygon_in.exterior.coords[-1][0]], [polygon_in.exterior.coords[-1][1]]
    for i in range(len(polygon_in.exterior.coords)):
        x_list.append(polygon_in.exterior.coords[i][0])
        y_list.append(polygon_in.exterior.coords[i][1])
    plt.plot(x_list,y_list,'k-', linewidth=1)
    if len(polygon_out) > 0:
        for poly in polygon_out:
            x_list, y_list = [poly.exterior.coords[-1][0]], [poly.exterior.coords[-1][1]]
            for i in range(len(poly.exterior.coords)):
                x_list.append(poly.exterior.coords[i][0])
                y_list.append(poly.exterior.coords[i][1])
            plt.plot(x_list,y_list,'k-', linewidth=1)
    if loadFactor is not None:
        plt.title('Load factor: %.2f' % loadFactor.item())
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show(block = False)
    return fig
