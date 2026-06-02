# Reverse Engineering Strut-and-Tie Models
# ================================
# Example Dapped-end beam, Case B
# Karin Yu, 2026

# Example Case D from Journal Paper "Grammar-based generation of strut-and-tie models for designing reinforced concrete structures" by K. Yu, M. A. Kraus, E. Chatzi, W. Kaufmann and  2026

__author__ = 'Karin Yu'

# Define geometry
L = 1500.  # mm
h = 600.  # mm
h_o = 300.  # mm
e_o = 350. # mm
cnom = 50. # 70 mm # 80 for modification of Solution C
L_load = 150. # mm
t = 250. # mm

X_stirrup = 1212.5 # mm
X_cross = 775 # mm

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

example_name = 'DEB_SolB' # name of the example
# Parameters for 1st part of layout optimization
max_num_iter = 10
max_threshold_S = 0.00001
max_threshold_C = 0.0001


# Plotting geometry
geom = Polygon([(0, h_o), (e_o, h_o), (e_o, 0), (L+e_o-L_load, 0), (L+e_o-L_load, h), (0, h)])
geom_out = []

# Effective geometry
polygon_in = Polygon([(cnom, h_o+cnom),(e_o+cnom, h_o+cnom), (e_o+cnom, cnom), (L+e_o-L_load, cnom), (L+e_o-L_load, h-cnom), (cnom, h-cnom)])
polygon_out = []
add_x = [L_load,e_o+cnom, X_cross, X_stirrup, L-L_load]
add_y = []

# boundary conditions
dof_coords = np.array([[L+e_o-L_load, cnom], [L+e_o-L_load, h-cnom]])
dofs = [[0,1], [0,0]]
force_coords = np.array([[L_load, h_o+cnom]])
forces = [[0, 1]] # unit force

# set reinforcement layout
reinf_list = np.array([[L_load, h_o+cnom, X_cross, h_o+cnom, 2*16**2*np.pi/4], # corner horizontal tie
                        [e_o+cnom, cnom, e_o+cnom, h-cnom, 2*14**2*np.pi/4], # corner vertical tie
                       [e_o+cnom, cnom, X_cross, cnom, 2*(16**2*np.pi/4)], # 1. horizontal tie
                        [X_cross, cnom, X_cross, h_o+cnom, 2*(14**2*np.pi/4)], # small vertical tie
                        [X_cross, cnom, X_stirrup, cnom, 4*(18**2*np.pi/4)], # 2. horizontal tie
                       [X_stirrup, cnom, L+e_o-L_load, cnom, 2*(30**2*np.pi/4)], # 3. horizontal tie
                       [X_stirrup, cnom, X_stirrup, h-cnom, 2*(14**2*np.pi/4)]]) # vertical tie
