# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

# Load up the 3D point cloud data from the appropriate .npy file
relative_path = "point_clouds/point_cloud_data_V3/point_cloud_data_randDist\handTest_randDist=50000,numNeighbors=12.npy"
point_cloud_data = np.load('./' + relative_path) # make sure the relative_path ends in the file_name.npy format
num_points = len(point_cloud_data)
print("point cloud size = " + str(num_points))

# Creating dataset
x = np.zeros(num_points)
y = np.zeros(num_points)
z = np.zeros(num_points)

# Populate dataset
inner_points_index = 0
for i in range(num_points):
    x[inner_points_index] = point_cloud_data[i][0]
    y[inner_points_index] = point_cloud_data[i][1]
    z[inner_points_index] = point_cloud_data[i][2]
    inner_points_index += 1


# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating scatter plot
ax.scatter3D(x, y, z, color = "green")

# create lines for neighbors
for i in range(num_points):
    for neighbor in range(len(point_cloud_data[0]) - 3):
        if not math.isnan((point_cloud_data[i][neighbor+3])):
            neighbor_index = int(point_cloud_data[i][neighbor+3])
            ax.plot([point_cloud_data[i][0], point_cloud_data[neighbor_index][0]], [point_cloud_data[i][1], point_cloud_data[neighbor_index][1]], [point_cloud_data[i][2], point_cloud_data[neighbor_index][2]])



plt.title("3D Point Cloud")
 
# show plot
plt.show()