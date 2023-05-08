import pybullet as p
import time
from datetime import datetime
import numpy as np
import pybullet_data
import random
import Simulation_Variables as vars
import os.path

# center of mass = [0.80875437 1.02478381 0.17250621]
# mesh extents = [4.71387499 4.42177465 1.91436915]

# FORMAT OF CONTACT POINT IN NUMPY (5x3) --> [[contact_X, contact_Y, contact_Z], [normal_X, normal_Y, normal_Z], [shear1_X, shear1_Y, shear1_Z], [shear2_X, shear2_Y, shear2_Z], [normal_Force, shear1_Force, shear2_Force]]
# FORMAT OF OUTPUT FILE STORING CONTACT POINTS (ith elemenent represents i*dt time so size is (ix4x5x3)) --> ith element = [P1, P2, P3, P4]
simulation_data_contacts = np.full((vars.max_frames, vars.max_contacts, 5, 3), np.nan)

t = 0
simulation_running = True
physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(vars.dt)
logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "box_test_1.json")
#disable rendering during creation.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

shift = [0, -0.02, 0]
meshScale = [1, 1, 1]
object_urdf = "flatPlate.urdf"
boxStartPos = [vars.obj_x_offset, vars.obj_y_offset, vars.obj_z_offset]
boxStartOr = p.getQuaternionFromEuler([vars.obj_roll, vars.obj_pitch, vars.obj_yaw])
handID = p.loadURDF(object_urdf, boxStartPos, boxStartOr, globalScaling = 1.0)
# -0.80875437 1.02478381 0.17250621
# Calculate bounding box of URDF and reload URDF but this time scaled
obj_extents = p.getAABB(handID)
aabbMin = obj_extents[0]
aabbMax = obj_extents[1]
dim_bounding_box = [aabbMax[0] - aabbMin[0], aabbMax[1] - aabbMin[1], aabbMax[2] - aabbMin[2]]
center_mass = p.getLinkState(handID, 0)
print("Obj Extents = " + str(dim_bounding_box))
print("Obj COM = " + str(p.getLinkState(handID, 0)))
p.removeBody(handID)
scaling_factor = float(vars.obj_scaling_factor)/float(max(dim_bounding_box))
print("scaling factor = "+str(scaling_factor))

# FIX THIS
boxStartPos = [0, 0, 0]

handID = p.loadURDF(object_urdf, boxStartPos, boxStartOr, globalScaling = (scaling_factor))

# print("Axis + Angle --> " + str(p.getAxisAngleFromQuaternion(boxStartOr)))
# p.resetBasePositionAndOrientation(handID, [0, 0, 0], p.getQuaternionFromEuler([1,0,0]))

#visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius = 0.5)
#collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.5)
#sphereID = p.createMultiBody(baseMass=0.0,
                       #baseInertialFramePosition=[0, 0, 0],
                       #baseCollisionShapeIndex=collisionShapeId,
                       #baseVisualShapeIndex=visualShapeId,
                       #basePosition =  [0.0, 0.0, 0.0],
                       #useMaximalCoordinates=True)

visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[vars.box_width, vars.box_height, vars.box_depth])
collisionShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[vars.box_width, vars.box_height, vars.box_depth])
boxID = p.createMultiBody(baseMass=vars.big_M,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[0, 0, vars.init_height],
                      useMaximalCoordinates=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.stopStateLogging(logId)
p.setGravity(0, 0, vars.grav_acc)
p.resetDebugVisualizerCamera(cameraDistance = vars.camera_dist, cameraYaw=-90, cameraPitch=-45, cameraTargetPosition=[0,0,0])
# Change physicall/material behaviour of box and floor
p.changeDynamics(boxID, -1, vars.big_M, vars.coeff_lateral_friction_box, vars.coeff_spinning_friction_box, vars.coeff_rolling_friction_box, vars.coeff_restitution_box)
p.changeDynamics(handID, -1, 0, vars.coeff_lateral_friction_floor, vars.coeff_spinning_friction_floor, vars.coeff_rolling_friction_floor, vars.coeff_restitution_floor)

while(simulation_running):
  p.stepSimulation()

  # Give the cube some initial linear and angular velocity
  if(t < 3*vars.dt and t > vars.dt):
    p.resetBaseVelocity(boxID, [vars.init_force_mag * (random.random()-0.5), vars.init_force_mag * (random.random()-0.5), vars.init_force_mag * (random.random()-0.5)], [(random.random()-0.5), (random.random()-0.5), (random.random()-0.5)])
  
  contactPoints = p.getContactPoints()
  position_orientation = p.getBasePositionAndOrientation(boxID)
  #print("position and orientation (PyBullet) = " + str(position_orientation))
  #print("\ncontacts (PyBullet) = "+str(contactPoints))
  #print("\n number of contacts = "+str(len(contactPoints)))

  # Convert contact points from PyBullet List to Numpy Array
  contacts_at_this_instance = np.full((vars.max_contacts, 5, 3), np.nan)
  print("number of contacts = " + str(len(contactPoints)))
  for i in range(min(len(contactPoints), vars.max_contacts)):
    contacts_at_this_instance[i][:][:] = [[contactPoints[i][5][0], contactPoints[i][5][1], contactPoints[i][5][2]], [contactPoints[i][7][0], contactPoints[i][7][1], contactPoints[i][7][2]], [contactPoints[i][11][0], contactPoints[i][11][1], contactPoints[i][11][2]], [contactPoints[i][13][0], contactPoints[i][13][1], contactPoints[i][13][2]], [contactPoints[i][9], contactPoints[i][10], contactPoints[i][12]]]
  #print("contacts (numpy) = "+str(contacts_at_this_instance))

  # Add one instance of contact points to overall simulation data
  simulation_data_contacts[int(t/vars.dt)][:][:][:] = contacts_at_this_instance
  time.sleep(1./240.)
  t += vars.dt
  print("time = "+str(t))

  if (int(t/vars.dt) >= vars.max_frames):
    simulation_running = False
  if (position_orientation[0][2] < aabbMin[2]):
    simulation_running = False

print("END OF SIMULATION")
now = datetime.now()
time_code = now.strftime("%d_%m_%Y_%H_%M_%S")
np.save('./contact_point_data/contact_data_'+time_code+'.npy', simulation_data_contacts)