import numpy as np
import taichi as ti
import time
import random
import math
from datetime import datetime
import Simulation_Variables as vars
np.set_printoptions(threshold=np.inf)

ti.init(default_ip=ti.i64)  # Sets the default integer type to ti.i64
ti.init(default_fp=ti.f64)  # Sets the default floating-point type to ti.f64

# Import a particular contact point data set from the PyBullet simulations
contact_point_data = np.load('contact_point_data/contact_data_23_04_2023_19_31_33.npy')
object_lattice_data = np.load('point_clouds/point_cloud_data_V3/point_cloud_data_randDist/flatPlate_randDist=5000,numNeighbors=12.npy')
#object_lattice_data = np.load('point_clouds\point_cloud_data_V3/point_cloud_data_randDist/handTest_randDist=100000,numNeighbors=12.npy')
#contact_point_data\contact_data_23_04_2023_14_55_03.npy
#contact_point_data\contact_data_23_04_2023_14_55_30.npy
#contact_point_data\contact_data_23_04_2023_14_56_29.npy

# Define some critical parameters
num_neighbors = int(len(object_lattice_data[0])-3)
print("number of neighbors = " + str(num_neighbors))
num_points = len(object_lattice_data)
print("number of points = " + str(num_points))
num_frames = int(len(contact_point_data))
print("number of frames = " + str(num_frames))
num_frames = contact_point_data.shape[0]
print("num_frames", num_frames)
num_contacts = contact_point_data.shape[1]
print("num_contacts", num_contacts)
num_contact_guesses = 10
print("num_contacts_guesses", num_contacts)
avg_radius_multiplier = 3.0
print("avg_radius_multiplier", num_contacts)
element_mass = vars.sum_element_mass/num_points
print("element_mass", element_mass)
max_force = vars.max_force
print("max_force", max_force)
max_vel = vars.max_vel
print("max_force", max_vel)

# Initialize the Vulkan architecture for Taichi
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
#arch = ti.cuda
ti.init(arch=arch)

# Get current time code
now = datetime.now()
time_code = now.strftime("%d_%m_%Y_%H_%M_%S")

# Particles array format [x, y, z, Vx, Vy, Vz, Neighbor1, Neighbor2, ..., NeighborN]
particles = ti.Vector.field(6+num_neighbors, dtype = ti.f64, shape = (num_points))
# Particles_Display array format [x, y, z]
particles_display = ti.Vector.field(3, dtype = ti.f32, shape = (num_points))
# Particles_Initial array format [x, y, z]
particles_initial = ti.Vector.field(3, dtype = ti.f64, shape = (num_points))
# Contact points data - convert to taichi field for quicker processing
contact_points = ti.field(dtype = ti.f64, shape = (num_frames, num_contacts, 5, 3))
# contact_point_data = contact_point_data.astype(ti.f64)
contact_points.from_numpy(contact_point_data)
# create list for displaying position of contacts in animation
contact_points_display = ti.Vector.field(3, dtype = ti.f32, shape = num_contacts)
# create a list for the points that are found to be the closest to the contacts - for display only
contact_points_closest_display = ti.Vector.field(3, dtype = ti.f32, shape = num_contacts)

@ti.kernel
def initiate_particles(object_lattice_data: ti.types.ndarray()):
    # Add every point from numpy array to taichi field
    for i in particles:
        particles[i][0] = object_lattice_data[i, 0]
        particles[i][1] = object_lattice_data[i, 1]
        particles[i][2] = object_lattice_data[i, 2]
        particles[i][3] = ((ti.random(ti.f32) - 0.5)*vars.initial_particle_rand_vel) # Start either with random velocity or zero
        particles[i][4] = ((ti.random(ti.f32) - 0.5)*vars.initial_particle_rand_vel) # Start either with random velocity or zero
        particles[i][5] = ((ti.random(ti.f32) - 0.5)*vars.initial_particle_rand_vel) # Start either with random velocity or zero
        particles_display[i][0] = object_lattice_data[i, 0]
        particles_display[i][1] = object_lattice_data[i, 1]
        particles_display[i][2] = object_lattice_data[i, 2]
        for j in ti.static(range(num_neighbors)):
            particles[i][6+j] = object_lattice_data[i, 3+j]

        # Create a field to store the initial positions of the particles. Used later.
        particles_initial[i] = [object_lattice_data[i, 0], object_lattice_data[i, 1], object_lattice_data[i, 2]]

# Add numpy data to taichi field for particles
initiate_particles(object_lattice_data)

def initiate_neighbors():
    # Add every possible connection to an array (including duplicates)
    temporary_neighbor_array = np.zeros((num_points*num_neighbors, 2))
    for i in range(num_points):
        for j in range(num_neighbors):
            temporary_neighbor_array[i*num_neighbors+j] = [i, object_lattice_data[i][3+j]]
    print("Total Connections (Before Filter) = "+str(temporary_neighbor_array))
    # Edit so that the first index is always smaller
    temporary_neighbor_array_V2 = np.zeros((num_points*num_neighbors, 2))
    for i in range(len(temporary_neighbor_array)):
        index1 = temporary_neighbor_array[i,0]
        index2 = temporary_neighbor_array[i,1]
        if math.isnan(index1) or math.isnan(index2):
            temporary_neighbor_array[i, 0] = 0
            temporary_neighbor_array[i, 1] = 0
        else:
            if index1 > index2:
                temporary_neighbor_array[i][0] = index2
                temporary_neighbor_array[i][1] = index1
    # Remove duplicates and values out of range and make into integer. Also remove all [0, 0] rows
    neighbor_array = np.unique(temporary_neighbor_array, axis=0)
    neighbor_array = neighbor_array[~np.all(neighbor_array == 0, axis=1)]

    neighbor_array = neighbor_array.astype(int)

    return neighbor_array

# Create connections array without dublicates. Add to taichi field
total_connections = initiate_neighbors()
total_connections_length = len(total_connections)
print("Total Connections = "+str(total_connections))
print("Total Num of Connections = "+str(total_connections_length))

# Neighbour_Connections array format [[N_i1, N_j1], [N_i2, N_j2], ..., [N_i_fin, N_j_fin]]
neighbor_connections = ti.Vector.field(2, dtype = ti.i32, shape = total_connections_length)
neighbor_connections.from_numpy(total_connections)

@ti.kernel
def find_avg_length() -> ti.f32:
    summation = 0.0
    ti.loop_config(serialize=True)
    for i in (range(total_connections_length)):
            indx1 = neighbor_connections[i][0]
            indx2 = neighbor_connections[i][1]
            P1 = ti.Vector([particles_initial[indx1][0], particles_initial[indx1][1], particles_initial[indx1][2]])
            P2 = ti.Vector([particles_initial[indx2][0], particles_initial[indx2][1], particles_initial[indx2][2]])
            length = float(ti.math.distance(P1, P2))
            #print("bond length = ", length)

            summation += length
    return summation/total_connections_length
avg_bond_length = find_avg_length()
print("Average Bond Length = "+str(avg_bond_length))

# Particles_Between array format [[Neighbor1_x, Neighbor1_y, Neighbor1_z], ..., [NeighborN_x, NeighborN_y, NeighborN_z]]
particles_between = ti.Vector.field(3, dtype = ti.f32, shape = total_connections_length)
# Particles_Between_Color array format [[Neighbor1_R, Neighbor1_G, Neighbor1_B], ..., [NeighborN_R, NeighborN_G, NeighborN_B]]
particles_between_color = ti.Vector.field(3, dtype = ti.f32, shape = total_connections_length)

@ti.kernel
def initiate_particles_between():
    for i in neighbor_connections:
        index1 = neighbor_connections[i][0]
        index2 = neighbor_connections[i][1]
        particles_between[i][0] = 0.5*(particles_initial[index1][0]+particles_initial[index2][0])
        particles_between[i][1] = 0.5*(particles_initial[index1][1]+particles_initial[index2][1])
        particles_between[i][2] = 0.5*(particles_initial[index1][2]+particles_initial[index2][2])
        particles_between_color[i] = [0.0, 0.0, 0.0]

initiate_particles_between()

contacts_indx_prev = ti.field(dtype = ti.i32, shape = (num_contacts))
contacts_indx_prev.fill(num_points+2)
# contacts_indx_prev = [indx_contact1, indx_contact2, ... , indx_contactn]
rough_contacts_indx = ti.field(dtype = ti.i32, shape = (num_contacts, num_contact_guesses))
rough_contacts_indx.fill(num_points+2)
print("Rough Contacts Index Array -> ", rough_contacts_indx)
# rough_contacts_indx = [[indx_contact1_guess1, indx_contact1_guess2 ... indx_contact1_guess5], [...], ... , [indx_contactn_guess1, ...indx_contactn_guess5]]

@ti.kernel
def update_points(time: ti.i32):
    # contacts_data = the contacts [n x 5 x 3] of the current frame at t
    # contacts_data_prev = the contacts [n x 5 x 3] of the frame at t-1
    # contacts_data_prev = the list of exact contact indices at t-1. The indecies match up with the particles array
    # OUTPUT = a "rough" list of points that are somewhat close to the frame t contacts

    # Below line should only be uncommented for testing reasons
    #ti.loop_config(serialize=True)
    for i in particles:
        applied_force = [0.0, 0.0, 0.0]

        gravity = [0.0, 0.0, element_mass*vars.grav_acc]
        if vars.gravity_on == False:
            gravity[2] = 0.0

        # Calculate external forces (a.k.a forces from contact points)
        external_force = [0.0, 0.0, 0.0]

        # Only for testing reasons we add sinusoidal input
        origin = ti.Vector([0.0, 0.0, 0.0])
        P2 = ti.Vector([particles[int(i)][0], particles[int(i)][1], particles[int(i)][2]])
        dist_to_origin = float(ti.math.distance(origin, P2))
        if dist_to_origin < 0.1 and time <= num_frames:
            s = 0
            #external_force[0] = 10*ti.sin(time*0.1)
            #external_force[1] = 10*ti.cos(time*0.1)
            external_force[2] = 100.0*ti.cos(time*0.8)

        if time <= num_frames:
            for j in (range(num_contacts)):
                contact_indx = int(contacts_indx_prev[j])
                if int(i) == contact_indx:
                    # some x-component of the force from the simulation file turned into taichi coordinates [NEED vector projection helper funcs]
                    if (not isnan(contact_points[time - 1, j, 1, 0])):
                        force_x = -vars.force_input_mult * (contact_points[time - 1, j, 1, 0]*contact_points[time - 1, j, 4, 0] + contact_points[time - 1, j, 2, 0]*contact_points[time - 1, j, 4, 1] + contact_points[time - 1, j, 3, 0]*contact_points[time - 1, j, 4, 2])
                        force_y = vars.force_input_mult * (contact_points[time - 1, j, 1, 1]*contact_points[time - 1, j, 4, 0] + contact_points[time - 1, j, 2, 1]*contact_points[time - 1, j, 4, 1] + contact_points[time - 1, j, 3, 1]*contact_points[time - 1, j, 4, 2])
                        force_z = vars.force_input_mult * (contact_points[time - 1, j, 1, 2]*contact_points[time - 1, j, 4, 0] + contact_points[time - 1, j, 2, 2]*contact_points[time - 1, j, 4, 1] + contact_points[time - 1, j, 3, 2]*contact_points[time - 1, j, 4, 2])
                        external_force[0] += force_x
                        print("force X = ", force_x)
                        external_force[1] += force_y
                        external_force[2] += force_z

                next_contact_pos = ti.Vector([contact_points[time, j, 0, 0], contact_points[time, j, 0, 1], contact_points[time, j, 0, 2]])
                current_point_pos = ti.Vector([particles[i][0], particles[i][1], particles[i][2]])
                dist_to_next_contact = float(ti.math.distance(next_contact_pos, current_point_pos))
            # print("Distance to Next Contact (Heuristic) = "+str(dist_to_next_contact))
                if(dist_to_next_contact <= avg_bond_length*avg_radius_multiplier):
                    rough_contacts_indx[j, ti.cast(ti.random(ti.f32)*num_contact_guesses, ti.i32)] = int(i)

        neighbor_forces = [0.0, 0.0, 0.0]
        index1 = int(i)
        for j in range(num_neighbors):
            index2 = (particles[i][6+j])
            if not isnan(index2):
                print("ith neighbor of "+str(index1)+" is "+str(index2))

                P1_i = ti.Vector([particles_initial[index1][0], particles_initial[index1][1], particles_initial[index1][2]])
                P2_i = ti.Vector([particles_initial[int(index2)][0], particles_initial[int(index2)][1], particles_initial[int(index2)][2]])
                rest_length = float(ti.math.distance(P1_i, P2_i))

                P1 = ti.Vector([particles[index1][0], particles[index1][1], particles[index1][2]])
                P2 = ti.Vector([particles[int(index2)][0], particles[int(index2)][1], particles[int(index2)][2]])
                current_length = float(ti.math.distance(P1, P2))

                lengths = [rest_length, current_length]
                print("lengths", lengths)

                neighbor_forces[0] += ((P2.x - P1.x)/(current_length))*(rest_length - current_length)*vars.k_Spring
                neighbor_forces[1] += ((P2.y - P1.y)/(current_length))*(rest_length - current_length)*vars.k_Spring
                neighbor_forces[2] += ((P2.z - P1.z)/(current_length))*(rest_length - current_length)*vars.k_Spring

                # Extra tangential force for rotation between points tto obey initial bond direction
                neighbor_initial = (P2_i - P1_i)
                neighbor_now = (P2 - P1)
                delta_position = neighbor_initial - neighbor_now
                proj_delta_onto_original_axis = ((ti.math.dot(delta_position, neighbor_now)/(ti.math.length(neighbor_now)*ti.math.length(neighbor_now)))*neighbor_now)
                tangential_component = delta_position - proj_delta_onto_original_axis

                neighbor_forces[0] += tangential_component.x*vars.k_Spring
                neighbor_forces[1] += tangential_component.y*vars.k_Spring
                neighbor_forces[2] += tangential_component.z*vars.k_Spring

                # IMPORTANT -> must also add a force to the other particle (ever action has opposite equal reaction)
                # IMPORTANT -> might have to fix something  here. Currently if I am your neighbor and you are mine you have two springs connecting you
                are_they_also_my_neighbor = False
                for k in range(num_neighbors):
                    if int(particles[int(index2)][6+k]) == index1:
                        are_they_also_my_neighbor = True

                if not are_they_also_my_neighbor:
                    neighbor_forces_prime = [0.0, 0.0, 0.0]
                    neighbor_forces_prime[0] += ((P1.x - P2.x)/(current_length))*(rest_length - current_length)*vars.k_Spring
                    neighbor_forces_prime[1] += ((P1.y - P2.y)/(current_length))*(rest_length - current_length)*vars.k_Spring
                    neighbor_forces_prime[2] += ((P1.z - P2.z)/(current_length))*(rest_length - current_length)*vars.k_Spring

                    particles[int(index2)][3] += vars.dt*neighbor_forces_prime[0]/element_mass
                    particles[int(index2)][4] += vars.dt*neighbor_forces_prime[1]/element_mass
                    particles[int(index2)][5] += vars.dt*neighbor_forces_prime[2]/element_mass

                # Cap the magnitude of each neightbor force component to max_force
                if neighbor_forces[0] > max_force:
                    neighbor_forces[0] = max_force
                elif neighbor_forces[0] < -max_force:
                    neighbor_forces[0] = -max_force

                if neighbor_forces[1] > max_force:
                    neighbor_forces[1] = max_force
                elif neighbor_forces[1] < -max_force:
                    neighbor_forces[1] = -max_force

                if neighbor_forces[2] > max_force:
                    neighbor_forces[2] = max_force
                elif neighbor_forces[2] < -max_force:
                    neighbor_forces[2] = -max_force

        applied_force[0] = gravity[0] + external_force[0] + neighbor_forces[0]
        applied_force[1] = gravity[1] + external_force[1] + neighbor_forces[1]
        applied_force[2] = gravity[2] + external_force[2] + neighbor_forces[2]

        # print("applied_force", applied_force)

        # Dampen velocities
        particles[i][3] *= vars.damping
        particles[i][4] *= vars.damping
        particles[i][5] *= vars.damping
        # Update velocities
        particles[i][3] += vars.dt*applied_force[0]/element_mass
        particles[i][4] += vars.dt*applied_force[1]/element_mass
        particles[i][5] += vars.dt*applied_force[2]/element_mass
        # Cap the magnitude of each velocity component to max_vel
        if particles[i][3] > max_vel:
            particles[i][3] = max_vel
        elif particles[i][3] < -max_vel:
            particles[i][3] = -max_vel

        if particles[i][4] > max_vel:
            particles[i][4] = max_vel
        elif particles[i][4] < -max_vel:
            particles[i][4] = -max_vel

        if particles[i][5] > max_vel:
            particles[i][5] = max_vel
        elif particles[i][5] < -max_vel:
            particles[i][5] = -max_vel

        # Update positions
        particles[i][0] += vars.dt*particles[i][3]
        particles[i][1] += vars.dt*particles[i][4]
        particles[i][2] += vars.dt*particles[i][5]

        particles_display[i][0] = particles[i][0]
        particles_display[i][1] = particles[i][1]
        particles_display[i][2] = particles[i][2]
    
    
    

@ti.kernel
def update_points_between():
    for i in neighbor_connections:
        # Get index of neighbors
        index1 = neighbor_connections[i][0]
        index2 = neighbor_connections[i][1]

        # Average of positions is center point
        particles_between[i][0] = 0.5*(particles[index1][0]+particles[index2][0])
        particles_between[i][1] = 0.5*(particles[index1][1]+particles[index2][1])
        particles_between[i][2] = 0.5*(particles[index1][2]+particles[index2][2])

        # Find original length and current length
        P1 = ti.Vector([particles_initial[index1][0], particles_initial[index1][1], particles_initial[index1][2]])
        P2 = ti.Vector([particles_initial[index2][0], particles_initial[index2][1], particles_initial[index2][2]])
        rest_length = float(ti.math.distance(P1, P2))

        P1 = ti.Vector([particles[index1][0], particles[index1][1], particles[index1][2]])
        P2 = ti.Vector([particles[index2][0], particles[index2][1], particles[index2][2]])
        current_length = float(ti.math.distance(P1, P2))

        # Calculate and assign color
        index = (current_length/rest_length) - 1.0
        # print("color index", index)
        
        color = ti.Vector([0.0, 0.0, 0.0])
        color_sensitivity = vars.color_sensitivity
        if(index < 0.0):
            color = [1.0, 1.0 + index*color_sensitivity, 1.0 + index*color_sensitivity]
        else:
            color = [1.0 - index*color_sensitivity, 1.0 - index*color_sensitivity, 1.0]
        particles_between_color[i] = color

@ti.kernel
def update_contacts_display(time: ti.i32):
    for i in range(num_contacts):
            position = (contact_points[time, i, 0, 0], contact_points[time, i, 0, 1], contact_points[time, i, 0, 2])
            if isnan(contact_points[time, i, 0, 0]):
                contact_points_display[i] = (2000.0, 2000.0, 2000.0)
            else:
                contact_points_display[i] = position

@ti.kernel
def find_closest_from_heuristic(time: ti.i32):
    print("rough contact guesses = " + str(rough_contacts_indx))
    ti.loop_config(serialize=True)
    for i in range(num_contacts):
        best_indx = num_points+2
        best_score = float('inf')
        all_are_nan = True
        for j in range(num_contact_guesses):
            current_indx = rough_contacts_indx[i, j]
            if (current_indx > num_points):
                nothing = 0.0
            else:  
                all_are_nan = False
                contact_coord = ti.Vector([contact_points[time, i, 0, 0], contact_points[time, i, 0, 1], contact_points[time, i, 0, 2]])
                rough_guess_coord = ti.Vector([particles[current_indx][0], particles[current_indx][1], particles[current_indx][2]])
                current_indx_dist = ti.math.distance(contact_coord, rough_guess_coord)
                if current_indx_dist <= best_score:
                    best_indx = current_indx
                    best_score = current_indx_dist

        if all_are_nan:
            contacts_indx_prev[i] = num_points+2
            contact_points_closest_display[i][0] = 2000.0
            contact_points_closest_display[i][1] = 2000.0
            contact_points_closest_display[i][2] = 2000.0
        else:
            best_indx_int = ti.cast(best_indx, ti.i32)
            contact_points_closest_display[i][0] = particles[best_indx_int][0]
            contact_points_closest_display[i][1] = particles[best_indx_int][1]
            contact_points_closest_display[i][2] = particles[best_indx_int][2]
            contacts_indx_prev[i] = best_indx_int

@ti.func
def minim(a: ti.f64, b: ti.f64) -> ti.f64:
    output = a
    if a > b:
        output = b
    return output


def main():

    # Create and run the taichi window and graphical environment
    window = ti.ui.Window("Taichi FEA Spring Simulation", (1200, 1000), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.5, 0.5, 0.5))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.lookat(0, 0, 0)
    canvas.scene(scene)
    t = 0

    while window.running and t < (vars.max_frames-1):
        t += 1
        print("Frame = "+str(t))
        ti.sync()
        camera.position(0.0, vars.camera_dist, vars.camera_dist)
        camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(-5, 0, 0), color=(1, 1, 1))
        scene.point_light(pos=(5, 0, 0), color=(1, 1, 1))
        scene.point_light(pos=(0, -20, 10), color=(1, 1, 1))

        # Update points with current contacts
        update_points(t)
        update_points_between()
        find_closest_from_heuristic(t-1)
        rough_contacts_indx.fill(num_points+2)
        print("Current index list of closest points to contact point = "+str(contacts_indx_prev))
        update_contacts_display(t)

        scene.particles(particles_between, radius = vars.particle_radius, per_vertex_color=particles_between_color)
        #scene.particles(contact_points_display, radius = vars.particle_radius*2, color=(1.0,0.0,0.0))
        #scene.particles(contact_points_closest_display, radius = vars.particle_radius*2, color=(0.0,1.0,0.0))
        #scene.particles(particles_display, radius = vars.particle_radius*2, color=(0.0,1.0,1.0))
        canvas.scene(scene)
        window.show()

@ti.func
def calc_dist(pos1, pos2):
    return ti.sqrt((pos2[0] - pos1[0])*(pos2[0] - pos1[0]) + (pos2[1] - pos1[1])*(pos2[1] - pos1[1]) + (pos2[2] - pos1[2])*(pos2[2] - pos1[2]))

@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

if __name__ == '__main__':
    main()



# TO DO LIST
# 1) create, save, and load a numpy array for positions of particles and connections for a 3D object
# 2) initialize the taichi vector fields (1D arrays in the form [X, Y, Z, V_X, V_Y, V_Z, A_X, A_Y, A_Z, neighbor1, neighbor2, neighbor3, neighbor4, neighbor 5, neighbor6])
# 3) instead of looping through points, try looping through connections

  #k_spring : -75 FOR 30x30x30
  #damping : 0.98
  #sum_element_mass : 0.4
  #dt : 0.001
  #force_input_mult : 0.0
  #initial_particle_rand_vel : 0.0
  #max_force : 50000
  #max_vel : 15000

#k_spring : -7 FOR 50x50x50
#damping : 0.95
#sum_element_mass : 0.4
#dt : 0.001
#force_input_mult : 1
#max_force : 50000
#max_vel : 2
