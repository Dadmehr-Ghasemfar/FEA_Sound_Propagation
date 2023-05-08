import numpy as np
import trimesh
import time
from datetime import datetime
import random
import math
import pyglet
import Simulation_Variables as vars
from trimesh import creation, transformations

mesh = trimesh.load('./flatPlate_V2_vhacd.obj')
print("Is mesh watertight ---> " + str(mesh.is_watertight))
print("Volume ratio of convex hull vs actual shape = " + str(mesh.volume / mesh.convex_hull.volume)+"\n")
# since the mesh is watertight, it means there is a
# volumetric center of mass which we can set as the origin for our mesh
print("center of mass = "+str(mesh.center_mass))
mesh.vertices -= mesh.center_mass

# Rotate our object by our universally defined roll-pitch-yaw and also scale by universal scaling
center = [0, 0, 0]
rot_matrix = transformations.rotation_matrix(vars.obj_rot_angle, vars.obj_rot_dir, center)
mesh.apply_transform(rot_matrix)
print("mesh extents = " + str((mesh.extents)))
scaling_factor = vars.obj_scaling_factor/max(mesh.bounding_box.extents)
mesh.apply_scale(scaling_factor)
print("scaling factor = " + str(scaling_factor))

# Create the lattice points in the format point = [x, y, z, neighbor1, neighbor2,..., neighbor_num_Neighbors] with num_Neighbors neighbors
points_coords = []
for i in range(vars.num_Lattice):
    x = random.random() * (vars.width) - (vars.width/2)
    y = random.random() * (vars.height) - (vars.height/2)
    z = random.random() * (vars.depth) - (vars.depth/2)
    #print("[x, y, z] = "+str([x, y, z]))
    print("percent done --> " + str((100*i/vars.num_Lattice)))
    if np.sign(trimesh.proximity.signed_distance(mesh, [[x, y, z]])[0]) == 1:
        points_coords.append([x, y, z])

# use the scipy.spatial library to create a distance matrix for all the points
from scipy.spatial import distance
D = distance.squareform(distance.pdist(points_coords))

# argsort row by row to get a list of closest points for each point
closest = np.argsort(D, axis=1)

# select num_Neighbors closest neighbors (make sure to disregard distance to self)
neighbor_indeces = (closest[:, 1:vars.num_Neighbors+1])   
points = np.append(points_coords, neighbor_indeces, 1)

sum = 0.0
index = 0
for index1 in range(points.shape[0]):
    for i in range(vars.num_Neighbors):
        index2 = int(points[index1, i+3])
        P1 = [points[index1, 0], points[index1, 1], points[index1, 2]]
        P2 = [points[index2, 0], points[index2, 1], points[index2, 2]]
        distance_points = math.dist(P1, P2)
        sum += distance_points
        index += 1
avg_bond_length = sum/index
print("average bond length = "+str(avg_bond_length))

output = np.zeros([points.shape[0], points.shape[1]])
for index1 in range(points.shape[0]):
    neighbor_index = np.zeros(vars.num_Neighbors)
    for i in range(vars.num_Neighbors):
        index2 = int(points[index1, i+3])
        P1 = [points[index1, 0], points[index1, 1], points[index1, 2]]
        P2 = [points[index2, 0], points[index2, 1], points[index2, 2]]
        distance_points = math.dist(P1, P2)

        if distance_points > vars.bond_cutoff*avg_bond_length:
            print("distance_points at index "+str([index1, i+3])+" is "+str(distance_points))
            points[index1, i+3] = float('nan')
            neighbor_index[i] = float('nan')
        else:
            neighbor_index[i] = index2
    output[index1] = np.concatenate((np.array(P1), neighbor_index))
print("output = "+str(output))

now = datetime.now()
time_code = now.strftime("%d_%m_%Y_%H_%M_%S")
np.save("./point_clouds/point_cloud_data_V3/point_cloud_data_randDist/flatPlate_randDist="+str(vars.num_Lattice)+",numNeighbors="+str(vars.num_Neighbors)+".npy", output)

# preview mesh in an opengl window if you installed pyglet and scipy with pip
mesh.show()

# IDEA FOR POINT NPY FORMAT [X, Y, Z, IN_OR_OUT, INDX NEIGHBOR1, INDX NEIGHBOR2, ..., INDX NEIGHBOR6] (if no neighbore --> NaN)
