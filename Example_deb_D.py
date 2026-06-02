# Reverse Engineering Strut-and-Tie Models
# ================================
# Example Dapped-end beam, Case D
# Karin Yu, 2026

# Example Case C from Journal Paper "Guided generation of strut-and-tie models for reinforced concrete structures with parametric graph grammatical evolution" by K. Yu, E. Chatzi, W. Kaufmann and M. A. Kraus 2026

__author__ = 'Karin Yu'

# Define geometry
L = 1500.  # mm
h = 600.  # mm
h_o = 300.  # mm
e_o = 350. # mm
cnom = 50. # 70 mm # 80 for modification of Solution C
L_load = 150. # mm
t = 250. # mm

# Define mesh.py size
mesh_fine = 50 # mm
mesh_coarse = 150 # mm

# Other material parameters
fcd = 20. # MPa
fsd = 435. # MPa
Es = 200000. # MPa
as_min = 4*8**2*np.pi/4 # mm^2
ac_min = t*mesh_coarse*1. # mm^2
max_incl = 3.
# final load factor depends on how much minimum reinforcement is accounted for!

example_name = ('DEB_SolD') # name of the example
# Parameters for 1st part of layout optimization
max_num_iter = 10
max_threshold_S = 0.00001
max_threshold_C = 0.0001

X_diag = e_o+cnom*4
X_diag_hor = 2*e_o+2*cnom
Y_diag = h_o-2*cnom

# Plotting geometry
geom = Polygon([(0, h_o), (e_o, h_o), (e_o, 0), (L+e_o-L_load, 0), (L+e_o-L_load, h), (0, h)])
geom_out = []

# Effective geometry
polygon_in = Polygon([(cnom, h_o+cnom),(e_o+cnom, h_o+cnom), (e_o+cnom, cnom), (L+e_o-L_load, cnom), (L+e_o-L_load, h-cnom), (cnom, h-cnom)])
polygon_out = []
add_x = [L_load,X_diag, X_diag_hor, L-L_load]
add_y = [Y_diag]

# boundary conditions
dof_coords = np.array([[L+e_o-L_load, cnom], [L+e_o-L_load, h-cnom]])
dofs = [[0,1], [0,0]]
force_coords = np.array([[L_load, h_o+cnom]])
forces = [[0, 1]] # unit force

# set reinforcement layout
reinf_list = np.array([[L_load, h-cnom, X_diag, Y_diag, 20**2*np.pi/4*2], # 1. diagonal tie
                        [X_diag, Y_diag, X_diag_hor, cnom, 22**2*np.pi/4*2], # 2. diagonal tie
                       [X_diag_hor, cnom, L-L_load, cnom, 2*(30**2*np.pi/4)], # 1. horizontal tie
                       [L-L_load, cnom, L-L_load+e_o, cnom, 2*(30**2*np.pi/4)], # 2. horizontal tie
                       [L-L_load, cnom, L-L_load, h-cnom, 2*(16**2*np.pi/4)]]) # vertical tie
