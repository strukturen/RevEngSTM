# Reverse Engineering Strut-and-Tie Models
# ================================
# Example Deep Beam with an Opening
# Karin Yu, 2026
# Data from "Toward a consistent design of structural concrete" by J. Schlaich, K. Schäfer and M. Jennewein 1987

# Rebar conversion
# #4 = 12.7 mm = 126 mm2
# #5 = 15.9 mm = 198 mm2
# #7 = 22.2 mm = 387 mm2
__author__ = 'Karin Yu'

# Define geometry
L = 7500.  # mm
h = 4700.  # mm
h_o = 1500.  # mm
e_o = 500. # mm
cnom = 50. # 70 mm # 80 for modification of Solution C
L_load = 4500.+250 # mm
t = 400. # mm

# reinforcement location
X_Reinf = e_o+h_o+1650.
D_Diag = 3900.
D_Cross = 2250.

# Define mesh.py size
mesh_fine = 50 # mm
mesh_coarse = 250 # mm

# Other material parameters
fcd = 17. # MPa
fsd = 434. # MPa
Es = 200000. # MPa
as_min = 4*10**2*np.pi/4 # mm^2
ac_min = t*mesh_coarse # mm^2, normal: t*mesh_coarse, for double: t*mesh_coarse*2.
max_incl = 3.
# final load factor depends on how much minimum reinforcement is accounted for!

# A = diagonal
# B = cross (adjust reinforcement)
# C = both
if len(add_args) > 0:
    case = add_args
else:
    case = 'B'
print('Case: ', case)
example_name = 'DeepBeam_Sol_Schlaich_'+case+'' # name of the example, _largerAc
# Parameters for 1st part of layout optimization
max_num_iter = 25
max_threshold_S = 0.00001
max_threshold_C = 0.0001

# Plotting geometry
geom = Polygon([(0, 0), (L, 0), (L, h), (0, h)])
geom_out = [Polygon([(e_o, e_o), (e_o + h_o, e_o), (e_o + h_o, e_o + h_o),
                        (e_o, e_o + h_o)])]

# Effective geometry
polygon_in = Polygon([(cnom, cnom), (L-cnom, cnom), (L-cnom, h-cnom), (cnom, h-cnom)])
polygon_out = [Polygon([(e_o-cnom, e_o-cnom), (e_o + h_o + cnom, e_o - cnom),(e_o + h_o + cnom,e_o + h_o + cnom), (e_o -cnom, e_o + h_o + cnom)])]
add_x = [e_o/2, D_Cross/2+e_o/4, D_Cross, D_Diag, h-cnom, L_load, L-e_o/2]
add_y = [e_o/2,  D_Cross/2+e_o/4, D_Cross, D_Diag]

# boundary conditions
dof_coords = np.array([[e_o/2., e_o/2], [L-e_o/2, e_o/2]])
dofs = [[0,0], [1,0]]
force_coords = np.array([[L_load, h-cnom]])
forces = [[0,-1]] # unit force

# set reinforcement layout
if case == 'A':
    reinf_list = np.array([[D_Diag, e_o/2, L-cnom, e_o/2, 4*387], # horizontal tie at the bottom, right (both parts) - if both then 2*4*387
                       [e_o/2, D_Diag, D_Diag, e_o/2, 4*387]]) # diagonal tie (diagonal)
elif case == 'B':
    reinf_list = np.array([[D_Diag, e_o/2, L-cnom, e_o/2, 4*387], # horizontal tie at the bottom, right (both parts) - if both then 2*4*387
                       [e_o/2, e_o/2, D_Diag, e_o/2, 4*387], # horizontal tie at the bottom, left (cross)
                       [e_o/2,  D_Cross, h-5*cnom, D_Cross, 2*7*198], # horizontal tie at the corner (cross)
                       [D_Cross, cnom, D_Cross, h-5*cnom, 2*7*198], # vertical tie at the corner (cross)
                       [e_o+h_o+cnom,  D_Cross/2+e_o/4, h-5*cnom, D_Cross/2+e_o/4, 2*5*126], # horizontal tie between the corner (cross)
                       [D_Cross/2+e_o/4, e_o+h_o+cnom, D_Cross/2+e_o/4, h-5*cnom, 2*5*126]]) # vertical tie between the corner (cross)
elif case == 'C':
    reinf_list = np.array([[D_Diag, e_o/2, L-cnom, e_o/2, 2*4*387], # horizontal tie at the bottom, right (both parts) - if both then 2*4*387
                       [e_o/2, D_Diag, D_Diag, e_o/2, 4*387], # diagonal tie (diagonal)
                       [e_o/2, e_o/2, D_Diag, e_o/2, 4*387], # horizontal tie at the bottom, left (cross)
                       [e_o/2,  D_Cross, h-5*cnom, D_Cross, 2*7*198], # horizontal tie at the corner (cross)
                       [D_Cross, cnom, D_Cross, h-5*cnom, 2*7*198], # vertical tie at the corner (cross)
                       [e_o+h_o+cnom,  D_Cross/2+e_o/4, h-5*cnom, D_Cross/2+e_o/4, 2*5*126], # horizontal tie between the corner (cross)
                       [D_Cross/2+e_o/4, e_o+h_o+cnom, D_Cross/2+e_o/4, h-5*cnom, 2*5*126]]) # vertical tie between the corner (cross)
