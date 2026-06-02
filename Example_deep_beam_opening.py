# Reverse Engineering Strut-and-Tie Models
# ================================
# Example Deep Beam with an Opening
# Karin Yu, 2026
# Data from "Example 4: Deep beam with opening" by Lawrence Novak and Heiko Sprenger in "Examples for the design of structural concrete with strut-and-tie models" by Karl-Heinz Reineck 2002

# Rebar conversion
# #4 = 12.7 mm = 126 mm2
# #5 = 15.875 mm = 198 mm2
# #7 = 22.225 mm = 387 mm2
# #8 = 25.4 mm = 506 mm2
# #10 = 32.258 mm = 817 mm2
__author__ = 'Karin Yu'

# Define geometry
L = 12700.  # mm
e_support = 350. # mm
h = 6000.  # mm
h_o = 2000.  # mm
w_o = 4000. # mm
e_ho = 2000. # mm
e_wo = 2000.+e_support # mm
cnom = 50.
L_load = 4000.+e_support # mm
h_reduced = 3500. # mm
L_reduced = 2000+4000+2250+e_support # mm
t = 305. # mm

# reinforcement location
Y_hor_1 = 500 #mm
Y_hor_2 = 4600 #mm

# Define mesh.py size
mesh_fine = 50 # mm
mesh_coarse = 300 # mm

# Other material parameters
fc = 31. # MPa
n_fc = (30/fc)**(1/3)
print(n_fc)
fcd = n_fc*fc/1.5 # MPa, design, n_t = 1
print(fcd)
fs = 414. # MPa
fsd = fs/1.15 # MPa, design
Es = 200000. # MPa
as_min = 126 # mm^2, 2-#4 at 18 inches = 1-#4
ac_min = t*mesh_coarse*1. # mm^2
max_incl = 3.
# final load factor depends on how much minimum reinforcement is accounted for!

example_name = 'DeepBeamOpening' # name of the example
# Parameters for 1st part of layout optimization
max_num_iter = 25
max_threshold_S = 0.00001
max_threshold_C = 0.001

# Plotting geometry
geom = Polygon([(0, 0), (L, 0), (L, h_reduced), (L_reduced, h_reduced), (L_reduced, h), (0, h)])
geom_out = [Polygon([(e_wo, e_ho), (e_wo +w_o, e_ho), (e_wo +w_o, e_ho + h_o),
                        (e_wo, e_ho + h_o)])]

# Effective geometry
polygon_in = Polygon([(cnom, 2*cnom), (L-cnom, 2*cnom), (L-cnom, h_reduced-cnom), (L_reduced-cnom, h_reduced-cnom), (L_reduced-cnom, h-2*cnom), (cnom, h-2*cnom)])
polygon_out = [Polygon([(e_wo-2*cnom, e_ho-2*cnom), (e_wo +w_o+2*cnom, e_ho-2*cnom), (e_wo +w_o+2*cnom, e_ho + h_o+cnom),
                        (e_wo-2*cnom, e_ho + h_o+cnom)])]
add_x = [e_support, L_load, L-1750, L-e_support]
add_y = [Y_hor_1, Y_hor_2]

# boundary conditions
dof_coords = np.array([[e_support, 2*cnom], [L-e_support, 2*cnom]])
dofs = [[0,0], [1,0]]
force_coords = np.array([[L_load, h-2*cnom]])
forces = [[0,-1]] # unit force

# set reinforcement layout
reinf_list = np.array([[cnom, h-2*cnom, L_reduced-cnom, h-2*cnom, 2*387], # top horizontal tie
                    [cnom, Y_hor_2, L_reduced-cnom, Y_hor_2, 2*4*198], # 2. horizontal tie above opening
                    [cnom, e_ho + h_o+cnom, L_reduced-cnom, e_ho + h_o+cnom, 4*817], # horizontal tie above opening
                    [cnom, e_ho - 2*cnom, L-cnom, e_ho - 2*cnom, 4*506], # horizontal tie below opening
                    [e_wo+w_o+2*cnom, h_reduced - cnom, L-cnom, h_reduced - cnom, 2*387], # horizontal tie at reduced end
                    [cnom, Y_hor_1, L-cnom, Y_hor_1, 2*4*198], # second bottom tie opening
                    [L-1750, cnom, L-1750, h_reduced-cnom, 2*9*126], # vertical tie at the reduced end
                    [cnom, 2*cnom, L-cnom, 2*cnom, 6*817]]) # bottom horizontal tie
