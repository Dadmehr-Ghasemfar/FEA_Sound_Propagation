import math
import pybullet as p
import yaml

with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)
(config_data['work_volume']['width'])
# Future Work --> Add a IsCartesianLattice = True and/or a IsRandDistLattice = true 
# and (combine the two process in one file)

# HAND OBJECT SETTINGS --> xyz = [200, 200, 200]
# DOVE OBJECT SETTINGS --> xyz = [20, 20, 10]
# COW  OBJECT SETTINGS --> xyz = [250, 150, 150]
width = (config_data['work_volume']['width'])
height = (config_data['work_volume']['height'])
depth = (config_data['work_volume']['depth'])

box_width = (config_data['box']['width'])
box_height =  (config_data['box']['height'])
box_depth = (config_data['box']['depth'])

big_M = (config_data['box']['mass'])
sum_element_mass = (config_data['simulation_vars']['sum_element_mass'])
init_force_mag = (config_data['init_conditions']['init_force_mag'])
init_height = (config_data['init_conditions']['init_height'])
k_Spring = (config_data['simulation_vars']['k_spring'])
damping = (config_data['simulation_vars']['damping'])
grav_acc = (config_data['simulation_vars']['grav_acc'])
force_input_mult = (config_data['simulation_vars']['force_input_mult'])
initial_particle_rand_vel = (config_data['simulation_vars']['initial_particle_rand_vel'])
max_force = (config_data['simulation_vars']['max_force'])
max_vel = (config_data['simulation_vars']['max_vel'])
gravity_on = (config_data['simulation_vars']['gravity_on'])

tension_threshold = (config_data['simulation_vars']['tension_threshold'])
color_sensitivity = (config_data['simulation_vars']['color_sensitivity'])
particle_radius = (config_data['simulation_vars']['particle_radius'])

# Assuming cartesian lattice point creation
xNum = (config_data['cartesian']['xNum'])
yNum = (config_data['cartesian']['yNum'])
zNum = (config_data['cartesian']['zNum'])

# Assuming random distribution lattice point creation
num_Lattice = (config_data['randDist']['num_Lattice'])
num_Neighbors = (config_data['randDist']['num_Neighbors'])
bond_cutoff = (config_data['randDist']['bond_cutoff'])

dt = (config_data['simulation_vars']['dt'])
max_frames = (config_data['simulation_vars']['max_frames'])
max_contacts = (config_data['simulation_vars']['max_contacts'])
camera_dist = (config_data['simulation_vars']['camera_dist'])

# COW OBJECT SETTINGS --> scale = 30, rpy = [pi/2,0,pi/2], offsets = [30, 0, 0]
# HAND OBJECT SETTINGS --> scale = 30, rpy = [-pi/2, -pi/4, -pi/4], offsets = [20, 50, 0]
# DOVE OBJECT SETTINGS --> scale = 200, rpy = [pi/2, 0, 0], offsets = [0, -30, 0]
obj_scaling_factor = (config_data['obj_parameters']['obj_scaling_factor'])
obj_roll = (config_data['obj_parameters']['obj_roll'])
obj_pitch = (config_data['obj_parameters']['obj_pitch'])
obj_yaw = (config_data['obj_parameters']['obj_yaw'])
obj_x_offset = (config_data['obj_parameters']['obj_x_offset'])
obj_y_offset = (config_data['obj_parameters']['obj_y_offset'])
obj_z_offset = (config_data['obj_parameters']['obj_z_offset'])
obj_rot_angle = p.getAxisAngleFromQuaternion(p.getQuaternionFromEuler([obj_roll, obj_pitch, obj_yaw]))[1]
obj_rot_dir = p.getAxisAngleFromQuaternion(p.getQuaternionFromEuler([obj_roll, obj_pitch, obj_yaw]))[0]

coeff_restitution_box = (config_data['box']['coeff_restitution_box'])
coeff_lateral_friction_box = (config_data['box']['coeff_lateral_friction_box'])
coeff_rolling_friction_box = (config_data['box']['coeff_rolling_friction_box'])
coeff_spinning_friction_box = (config_data['box']['coeff_spinning_friction_box'])
coeff_contact_damping_box = (config_data['box']['coeff_contact_damping_box'])
coeff_contact_stiffness_box = (config_data['box']['coeff_contact_stiffness_box'])

coeff_restitution_floor =  (config_data['obj_parameters']['coeff_restitution_floor'])
coeff_lateral_friction_floor = (config_data['obj_parameters']['coeff_lateral_friction_floor'])
coeff_rolling_friction_floor = (config_data['obj_parameters']['coeff_rolling_friction_floor'])
coeff_spinning_friction_floor = (config_data['obj_parameters']['coeff_spinning_friction_floor'])
coeff_contact_damping_floor = (config_data['obj_parameters']['coeff_contact_damping_floor'])
coeff_contact_stiffness_floor = (config_data['obj_parameters']['coeff_contact_stiffness_floor'])


