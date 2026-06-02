# Reverse Engineering Strut-and-Tie Models
# ================================
# Example Deep Beam with an Opening
# Karin Yu, 2026
# Data from "Example 8: High wall with two openings" by Robert W. Barnes in "Examples for the design of structural concrete with strut-and-tie models" by Karl-Heinz Reineck 2002

__author__ = 'Karin Yu'

# load custom source files
from src.problem import *
from src.plotter import *
from src.solver import *
from src.geom_optim import *

# Define geometry = dependent on inches
inch = 25.4 # mm
h = 640*inch # mm
w = 320*inch # mm
h_o = 120*inch # mm
e_support = round(20*inch,1) # mm
e_ho1 = 200*inch # mm
e_wo1 = 40*inch # mm
e_ho2 = 400*inch # mm
e_wo2 = 160*inch # mm
L_load_vertical = 100*inch # mm
L_load_horizontal1 = 360*inch # mm
L_load_horizontal2 = 620*inch # mm
cnom = round(10*inch,1) # mm
t = 16.*inch # mm

# reinforcement location
x_reinf1 = 140*inch # mm
x_reinf2 = 180*inch # mm
y_reinf1 = 185*inch # mm
y_reinf2 = 335*inch # mm
y_reinf3 = 385*inch # mm
y_reinf4 = 435*inch # mm
y_reinf5 = 535*inch # mm

# Rebar conversion
# #4 = 12.7 mm = 126 mm2
# #5 = 15.875 mm = 198 mm2
# #6 = 19.05 mm = 285 mm2
# #7 = 22.225 mm = 387 mm2
# #8 = 25.4 mm = 506 mm2
# #10 = 32.258 mm = 817 mm2
a_d4 = 126
a_d5 = 198
a_d6 = 285
a_d7 = 387
a_d8 = 506
a_d10 = 817

# Define mesh.py size
mesh_fine = 20*inch# mm
mesh_coarse = 40*inch # mm

print('Mesh size: %.2f mm' % mesh_coarse)
print('Small mesh.py size: %.2f mm' % mesh_fine)

print('### Start')

# Other material parameters, but with SIA Code conversion for f_cd
fc = 26. # MPa
n_fc = (30/fc)**(1/3)
fcd = n_fc*fc/1.5 # MPa, design, n_t = 1
fs = 410. # MPa
fsd = fs/1.15 # MPa, design
print('# Material properties\n# fcd = %f MPa, fsd = %f MPa' % (fcd, fsd))
Es = 200000. # MPa
as_min = 198 # 2* #5@12in = 304.8 mm = 1x#5
ac_min = t*mesh_coarse*.5 # mm^2
max_incl = 3.
# final load factor depends on how much minimum reinforcement is accounted for!

# Load Case 1: only vertical
# Load Case 2: horizontal from the right
# Load Case 3: LC1 + LC2 = right
# Load Case 4: horizontal from the left
# Load Case 5: LC1 + LC4 = left

if len(add_args) > 0:
    case = add_args
else:
    case = 'A'

example_name = 'Wall_2Open_Sol'+case
# Parameters for 1st part of layout optimization
max_num_iter = 25
max_threshold_S = 0.0001 # LC3: 0.0001
max_threshold_C = 0.001 # LC3: 0.001

# Plotting geometry
geom = Polygon([(0, 0), (w, 0), (w,h), (0, h)])
geom_out = [Polygon([(e_wo1, e_ho1), (e_wo1+h_o, e_ho1), (e_wo1+h_o, e_ho1+h_o), (e_wo1, e_ho1+h_o)]), Polygon([(e_wo2, e_ho2), (e_wo2+h_o, e_ho2), (e_wo2+h_o, e_ho2+h_o), (e_wo2, e_ho2+h_o)])]

# Effective geometry
polygon_in = Polygon([(cnom, cnom), (w-cnom, cnom), (w-cnom,h-cnom), (cnom, h-cnom)])
polygon_out = [Polygon([(e_wo1-cnom, e_ho1-cnom), (e_wo1+h_o+2*cnom, e_ho1-cnom), (e_wo1+h_o+2*cnom, e_ho1+h_o+cnom), (e_wo1-cnom, e_ho1+h_o+cnom)]), Polygon([(e_wo2-2*cnom, e_ho2-cnom), (e_wo2+h_o+cnom, e_ho2-cnom), (e_wo2+h_o+cnom, e_ho2+h_o+cnom), (e_wo2-2*cnom, e_ho2+h_o+cnom)])]

# additional points
add_x = [e_support, L_load_vertical,x_reinf1, x_reinf2, w-L_load_vertical,w-e_support]
add_y = [y_reinf1, (y_reinf1+y_reinf2)/2, y_reinf2, L_load_horizontal1, y_reinf3, y_reinf4, y_reinf5, L_load_horizontal2]

# boundary conditions
dof_coords = np.array([[e_support, cnom], [w - e_support, cnom]])
if case == 'A':
    dofs = [[0, 0], [1, 0]]  # B, left: [1,0], [0,0] A, right: [0,0], [1,0]
    force_coords = np.array([[L_load_vertical, h - cnom], [w - L_load_vertical, h - cnom], [w-cnom, L_load_horizontal1], [w-cnom, L_load_horizontal2]])  # B left: [cnom, L_load_horizontal1], [cnom, L_load_horizontal2]], A right: [w-cnom, L_load_horizontal1], [w-cnom, L_load_horizontal2]
    forces = [[0, -1], [0, -1], [-1 / 4.5 * 1.7, 0], [-1 / 4.5 * 1.7, 0]]  # unit force B left: +, A right: -
elif case == 'B':
        dofs = [[1, 0], [0, 0]]  # B, left: [1,0], [0,0] A, right: [0,0], [1,0]
        force_coords = np.array(
            [[L_load_vertical, h - cnom], [w - L_load_vertical, h - cnom], [cnom, L_load_horizontal1], [cnom,
                                                                                                            L_load_horizontal2]])  # B left: [cnom, L_load_horizontal1], [cnom, L_load_horizontal2]], A right: [w-cnom, L_load_horizontal1], [w-cnom, L_load_horizontal2]
        forces = [[0, -1], [0, -1], [1 / 4.5 * 1.7, 0], [1 / 4.5 * 1.7, 0]]  # unit force B left: +, A right: -

# set reinforcement layout
reinf_list = np.array([
    [x_reinf1, y_reinf2, x_reinf1, y_reinf4, 2 * 3 * a_d7],  # vertical tie middle top
    [x_reinf2, (y_reinf1 + y_reinf2) / 2, x_reinf2, y_reinf3, 2 * 7 * a_d7],  # vertical tie middle middle
    [x_reinf2, cnom, x_reinf2, (y_reinf1 + y_reinf2) / 2, 2 * 6 * a_d7],  # vertical tie middle bottom
    [e_support, y_reinf2, e_support, y_reinf3, 2 * 2 * a_d7],  # vertical tie right middle
    [w - e_support, cnom, w - e_support, y_reinf2, 2 * 11 * a_d7],  # vertical tie right first bottom
    [e_support, y_reinf4, e_support, y_reinf5, 2 * 9 * a_d5],  # vertical tie left top
    [e_support, y_reinf1, e_support, y_reinf4, 2 * 11 * a_d5],  # vertical tie left middle
    [e_support, cnom, e_support, y_reinf1, 2 * 11 * a_d7],  # vertical tie left first bottom
    [cnom, y_reinf5, w - cnom, y_reinf5, 2 * 8 * a_d6],  # horizontal above 2nd opening
    [cnom, y_reinf4, x_reinf1, y_reinf4, 2 * 7 * a_d5],  # horizontal next to 2nd opening
    [x_reinf1, y_reinf3, w - cnom, y_reinf3, 2 * 5 * a_d5],  # horizontal below 2nd opening
    [cnom, y_reinf2, w - cnom, y_reinf2, 2 * 7 * a_d7],  # horizontal above 1st opening
    [cnom, y_reinf1, w - cnom, y_reinf1, 2 * 7 * a_d7],  # horizontal below 1st opening
    [cnom, cnom, w - cnom, cnom, 2 * 3 * a_d7]])  # bottom horizontal tie
