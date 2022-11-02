import taichi as ti
import pybullet as p
import time
import math
import pybullet_data

k_Spring = -4
m = 10
d = 0.98
tension_threshold = 0
color_sensitivity = 1
particle_radius = 0.05

xNum = 50
yNum = 50
zNum = 5

width = 25
height = 25
depth = 3

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1. / 120.)
logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "box_test_1.json")
#disable rendering during creation.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

shift = [0, -0.02, 0]
meshScale = [0.1, 0.1, 0.1]
#the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
visualShapeId = p.createVisualShape(p.GEOM_PLANE)
collisionShapeId = p.createCollisionShape(p.GEOM_PLANE)
floorID = p.createMultiBody(baseMass=0,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[0, 0, depth/2],
                      useMaximalCoordinates=True)

visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 1])
collisionShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 1])
boxID = p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[0, 0, depth],
                      useMaximalCoordinates=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.stopStateLogging(logId)
p.setGravity(0, 0, -10)

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
#arch = ti.cuda
ti.init(arch=arch)

extremes_size = int(4 * xNum * yNum * zNum)
contact_points_size = int(xNum*yNum*0.25)

index_color = [0.0, 0.0, 0.0]
# FULLY FIXED CONDITIONS
TOP_FIXED = False
BOTTOM_FIXED = True

RIGHT_FIXED = False
LEFT_FIXED = False

BACK_FIXED = False
FRONT_FIXED = False

# DRIVEN CONDITIONS
RIGHT_DRIVEN = False
LEFT_DRIVEN = False

TOP_DRIVEN = False
BOTTOM_DRIVEN = False

FRONT_DRIVEN = False
BACK_DRIVEN = False

# NORMALLY FIXED CONDITIONS
TOP_NORMAL_FIXED = False
BOTTOM_NORMAL_FIXED = False

RIGHT_NORMAL_FIXED = False
LEFT_NORMAL_FIXED = False

BACK_NORMAL_FIXED = False
FRONT_NORMAL_FIXED = False

# video_manager = ti.tools.VideoManager(output_dir="./output", framerate=24, automatic_build=False)

particles_pos = ti.Vector.field(3, dtype = ti.f32, shape = (xNum * yNum * zNum))
particles_vel = ti.Vector.field(3, dtype = ti.f32, shape = (xNum * yNum * zNum))
particles_acc = ti.Vector.field(3, dtype = ti.f32, shape = (xNum * yNum * zNum))
particles_force_applied = ti.Vector.field(3, dtype = ti.f32, shape = (xNum * yNum * zNum))
particles_between_color = ti.Vector.field(3, dtype = ti.f32, shape = (3 * (xNum-0) * (yNum-0) * (zNum-0)))
particles_between = ti.Vector.field(3, dtype = ti.f32, shape = (3 * (xNum-0) * (yNum-0) * (zNum-0)))
particles_extremes = ti.Vector.field(3, dtype = ti.f32, shape = extremes_size)
particles_extremes_color = ti.Vector.field(3, dtype = ti.f32, shape = extremes_size)
contact_points = ti.Vector.field(8, dtype = ti.f32, shape = contact_points_size)

cube_corners_1 = ti.Vector.field(3, dtype = ti.f32, shape = 8)
cube_corners_2 = ti.Vector.field(3, dtype = ti.f32, shape = 8)
cube_center = ti.Vector.field(3, dtype = ti.f32, shape = 1)


@ti.kernel
def update_points(t: ti.f32):
    index_extremes = 0
    for i in particles_pos:
        x = x(i)
        y = y(i)
        z = z(i)
        k = k_Spring
        unit_dim = dx_dy_dz()
        force = ti.Vector([0.0, 0.0, 0.0])
        applied_force1 = ti.Vector([10*k*ti.sin(t), 0.0, 0.0])
        #particles_force_applied[i] = [0, 0, 0]

# INSIDE VOLUME
        if x != 0 and x != (xNum-1):
            if y != 0 and y != (yNum-1):
                if z != 0 and z != (zNum-1):
                    fz1 = lower_force(x, y, z)
                    fz2 = upper_force(x, y, z)
                    fx1 = lefter_force(x, y, z)
                    fx2 = righter_force(x, y, z)
                    fy1 = backer_force(x, y, z)
                    fy2 = fronter_force(x, y, z)
                    force = (fz1+fz2+fx1+fx2+fy1+fy2)
                    
                    #print(lower_force(x,y,z))
        else:
# LEFT AND RIGHT FACES
            if y != 0 and y != (yNum-1):
                if z != 0 and z != (zNum-1):
                    force += backer_force(x, y, z)
                    force += fronter_force(x, y, z)
                    force += upper_force(x, y, z)
                    force += lower_force(x, y, z)
                    if x == 0:
                        force += righter_force(x, y, z)
                        if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
                        if(LEFT_FIXED):
                            if LEFT_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
                    if x == (xNum-1):
                        force += lefter_force(x, y, z)
                        if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
                        if(RIGHT_FIXED):
                            if RIGHT_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
# FRONT AND REAR FACES
        if y == 0:
            if x != 0 and x != (xNum-1):
                if z != 0 and z != (zNum-1):
                    force += righter_force(x, y, z)
                    force += lefter_force(x, y, z)
                    force += upper_force(x, y, z)
                    force += lower_force(x, y, z)
                    force += backer_force(x, y, z)
                    if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
                    if(FRONT_FIXED):
                            if FRONT_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]

        if y == (yNum-1):
            if x != 0 and x != (xNum-1):
                if z != 0 and z != (zNum-1):
                    force += righter_force(x, y, z)
                    force += lefter_force(x, y, z)
                    force += upper_force(x, y, z)
                    force += lower_force(x, y, z)
                    force += fronter_force(x, y, z)
                    if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
                    if(BACK_FIXED):
                            if BACK_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
# TOP AND BOTTOM FACES
        if z == 0:
            if x != 0 and x != (xNum-1):
                if y != 0 and y != (yNum-1):
                    force += righter_force(x, y, z)
                    force += lefter_force(x, y, z)
                    force += backer_force(x, y, z)
                    force += fronter_force(x, y, z)
                    force += upper_force(x, y, z)
                    if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                    if(BOTTOM_FIXED):
                            if BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]

        if z == (zNum-1):
            if x != 0 and x != (xNum-1):
                if y != 0 and y != (yNum-1):
                    force += righter_force(x, y, z)
                    force += lefter_force(x, y, z)
                    force += backer_force(x, y, z)
                    force += fronter_force(x, y, z)
                    force += lower_force(x, y, z)
                    if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                    if(TOP_FIXED):
                            if TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
# BOTTOM HORIZONTAL EDGES
        if z == 0:
            if y != 0 and y != (yNum-1) and x == 0:
                force += backer_force(x, y, z)
                force += fronter_force(x, y, z)
                force += righter_force(x, y, z)
                force += upper_force(x, y, z)
                if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
                if(LEFT_FIXED or BOTTOM_FIXED):
                            if LEFT_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
            elif y != 0 and y != (yNum-1) and x == (xNum-1):
                force += backer_force(x, y, z)
                force += fronter_force(x, y, z)
                force += lefter_force(x, y, z)
                force += upper_force(x, y, z)
                if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
                if(RIGHT_FIXED or BOTTOM_FIXED):
                            if RIGHT_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
            elif x != 0 and x != (xNum-1) and y == 0:
                force += righter_force(x, y, z)
                force += lefter_force(x, y, z)
                force += backer_force(x, y, z)
                force += upper_force(x, y, z)
                if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
                if(FRONT_FIXED or BOTTOM_FIXED):
                            if FRONT_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
            elif x != 0 and x != (xNum-1) and y == (yNum-1):
                force += righter_force(x, y, z)
                force += lefter_force(x, y, z)
                force += fronter_force(x, y, z)
                force += upper_force(x, y, z)
                if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
                if(BACK_FIXED or BOTTOM_FIXED):
                            if BACK_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
# TOP HORIZONTAL EDGES
        elif z == (zNum-1):
            if y != 0 and y != (yNum-1) and x == 0:
                force += backer_force(x, y, z)
                force += fronter_force(x, y, z)
                force += righter_force(x, y, z)
                force += lower_force(x, y, z)
                if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
                if(LEFT_FIXED or TOP_FIXED):
                            if LEFT_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
            elif y != 0 and y != (yNum-1) and x == (xNum-1):
                force += backer_force(x, y, z)
                force += fronter_force(x, y, z)
                force += lefter_force(x, y, z)
                force += lower_force(x, y, z)
                if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
                if(RIGHT_FIXED or TOP_FIXED):
                            if RIGHT_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
            elif x != 0 and x != (xNum-1) and y == 0:
                force += righter_force(x, y, z)
                force += lefter_force(x, y, z)
                force += backer_force(x, y, z)
                force += lower_force(x, y, z)
                if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
                if(FRONT_FIXED or TOP_FIXED):
                            if FRONT_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
            elif x != 0 and x != (xNum-1) and y == (yNum-1):
                force += righter_force(x, y, z)
                force += lefter_force(x, y, z)
                force += fronter_force(x, y, z)
                force += lower_force(x, y, z)
                if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
                if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
                if(BACK_FIXED or TOP_FIXED):
                            if BACK_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
# VERTICAL EDGES
        if x == 0 and y == 0 and z != 0 and z != (zNum-1):
            force += upper_force(x, y, z)
            force += lower_force(x, y, z)
            force += righter_force(x, y, z)
            force += backer_force(x, y, z)
            if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(LEFT_FIXED or FRONT_FIXED):
                            if LEFT_DRIVEN or FRONT_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == (xNum-1) and y == 0 and z != 0 and z != (zNum-1):
            force += upper_force(x, y, z)
            force += lower_force(x, y, z)
            force += lefter_force(x, y, z)
            force += backer_force(x, y, z)
            if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(RIGHT_FIXED or FRONT_FIXED):
                            if RIGHT_DRIVEN or FRONT_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == 0 and y == (yNum-1) and z != 0 and z != (zNum-1):
            force += upper_force(x, y, z)
            force += lower_force(x, y, z)
            force += righter_force(x, y, z)
            force += fronter_force(x, y, z)
            if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(LEFT_FIXED or BACK_FIXED):
                            if LEFT_DRIVEN or BACK_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == (xNum-1) and y == (yNum-1) and z != 0 and z != (zNum-1):
            force += upper_force(x, y, z)
            force += lower_force(x, y, z)
            force += lefter_force(x, y, z)
            force += fronter_force(x, y, z)
            if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(RIGHT_FIXED or BACK_FIXED):
                            if RIGHT_DRIVEN or BACK_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]

# CORNER POINTS
        if x == 0 and y == 0 and z == 0:
            force += upper_force(x, y, z)
            force += backer_force(x, y, z)
            force += righter_force(x, y, z)
            if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(LEFT_FIXED or FRONT_FIXED or BOTTOM_FIXED):
                            if LEFT_DRIVEN or FRONT_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == 0 and y == 0 and z == (zNum-1):
            force += lower_force(x, y, z)
            force += backer_force(x, y, z)
            force += righter_force(x, y, z)
            if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
            if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(LEFT_FIXED or FRONT_FIXED or TOP_FIXED):
                            if LEFT_DRIVEN or FRONT_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == 0 and y == (yNum-1) and z == 0:
            force += upper_force(x, y, z)
            force += fronter_force(x, y, z)
            force += righter_force(x, y, z)
            if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(LEFT_FIXED or BACK_FIXED or BOTTOM_FIXED):
                            if LEFT_DRIVEN or BACK_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == 0 and y == (yNum-1) and z == (zNum-1):
            force += lower_force(x, y, z)
            force += fronter_force(x, y, z)
            force += righter_force(x, y, z)
            if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(LEFT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
            if(LEFT_FIXED or BACK_FIXED or TOP_FIXED):
                            if LEFT_DRIVEN or BACK_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == (xNum-1) and y == 0 and z == 0:
            force += upper_force(x, y, z)
            force += backer_force(x, y, z)
            force += lefter_force(x, y, z)
            if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
            if(RIGHT_FIXED or FRONT_FIXED or BOTTOM_FIXED):
                            if RIGHT_DRIVEN or FRONT_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == (xNum-1) and y == (yNum-1) and z == 0:
            force += upper_force(x, y, z)
            force += fronter_force(x, y, z)
            force += lefter_force(x, y, z)
            if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(BOTTOM_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
            if(RIGHT_FIXED or BACK_FIXED or BOTTOM_FIXED):
                            if RIGHT_DRIVEN or BACK_DRIVEN or BOTTOM_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == (xNum-1) and y == 0 and z == (zNum-1):
            force += lower_force(x, y, z)
            force += backer_force(x, y, z)
            force += lefter_force(x, y, z)
            if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
            if(FRONT_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(RIGHT_FIXED or FRONT_FIXED or TOP_FIXED):
                            if RIGHT_DRIVEN or FRONT_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]
        elif x == (xNum-1) and y == (yNum-1) and z == (zNum-1):
            force += lower_force(x, y, z)
            force += fronter_force(x, y, z)
            force += lefter_force(x, y, z)
            if(BACK_NORMAL_FIXED):
                                    particles_pos[i][1] = xyz_to_coor(x, y, z)[1]
                                    force[1] = 0.0
            if(RIGHT_NORMAL_FIXED):
                                    particles_pos[i][0] = xyz_to_coor(x, y, z)[0]
                                    force[0] = 0.0
            if(TOP_NORMAL_FIXED):
                                    particles_pos[i][2] = xyz_to_coor(x, y, z)[2]
                                    force[2] = 0.0
            if(RIGHT_FIXED or BACK_FIXED or TOP_FIXED):
                            if RIGHT_DRIVEN or BACK_DRIVEN or TOP_DRIVEN:
                                particles_pos[i] = xyz_to_coor(x+applied_force1[0], y+applied_force1[1], z+applied_force1[2])
                                force = [0.0, 0.0, 0.0]
                            else:
                                particles_pos[i] = xyz_to_coor(x, y, z)
                                force = [0.0, 0.0, 0.0]

        force += particles_force_applied[i]
        particles_acc[i] = force/m
        particles_vel[i] += particles_acc[i]
        particles_pos[i] += particles_vel[i]
        particles_vel[i] *= d

        if(x != 0 and y != 0 and z != 0):
            particles_between[i*3] = [ (0.5*particles_pos[i][0]+0.5*particles_pos[index(x-1, y, z)][0]), (0.5*particles_pos[i][1]+0.5*particles_pos[index(x-1, y, z)][1]), (0.5*particles_pos[i][2]+0.5*particles_pos[index(x-1, y, z)][2])]
            particles_between[i*3+1] = [ (0.5*particles_pos[i][0]+0.5*particles_pos[index(x, y-1, z)][0]), (0.5*particles_pos[i][1]+0.5*particles_pos[index(x, y-1, z)][1]), (0.5*particles_pos[i][2]+0.5*particles_pos[index(x, y-1, z)][2])]
            particles_between[i*3+2] = [ (0.5*particles_pos[i][0]+0.5*particles_pos[index(x, y, z-1)][0]), (0.5*particles_pos[i][1]+0.5*particles_pos[index(x, y, z-1)][1]), (0.5*particles_pos[i][2]+0.5*particles_pos[index(x, y, z-1)][2])]
            x_index = calc_dist(particles_pos[i], particles_pos[index(x-1,y,z)])/unit_dim[0]
            y_index = calc_dist(particles_pos[i], particles_pos[index(x,y-1,z)])/unit_dim[1]
            z_index = calc_dist(particles_pos[i], particles_pos[index(x,y,z-1)])/unit_dim[2]
            #print (x_index)
            c = color_sensitivity
            if x_index < 1:
                particles_between_color[i*3] = [1.0, c*x_index, c*x_index]
            else:
                particles_between_color[i*3] = [2.0 - c*x_index, 2.0 - c*x_index, 1.0]

            if y_index < 1:
                particles_between_color[i*3+1] = [1.0, y_index, y_index]
            else:
                particles_between_color[i*3+1] = [2.0 - c*y_index, 2.0 - c*y_index, 1.0]

            if z_index < 1:
                particles_between_color[i*3+2] = [1.0, z_index, z_index]
            else:
                particles_between_color[i*3+2] = [2.0 - c*z_index, 2.0 - c*z_index, 1.0]
            
            b = tension_threshold
            if (x_index < 1-b or x_index > 1+b):
                particles_extremes[i*3] = particles_between[i*3]
                particles_extremes_color[i*3] = particles_between_color[i*3]
                index_extremes += 1
            else:
                particles_extremes[i*3] = [0, 0, 0]
                particles_extremes_color[i*3] = [0, 0, 0]

            if (y_index < 1-b or y_index > 1+b):
                particles_extremes[i*3+1] = particles_between[i*3+1]
                particles_extremes_color[i*3+1] = particles_between_color[i*3+1]
                index_extremes += 1
            else:
                particles_extremes[i*3+1] = [0, 0, 0]
                particles_extremes_color[i*3+1] = [0, 0, 0]

            if (z_index < 1-b or z_index > 1+b):
                particles_extremes[i*3+2] = particles_between[i*3+2]
                particles_extremes_color[i+3+2] = particles_between_color[i*3+2]
                index_extremes += 1
            else:
                particles_extremes[i*3+2] = [0, 0, 0]
                particles_extremes_color[i*3+2] = [0, 0, 0]
    
    for j in range(contact_points_size):
        index_contact = contact_points[j][0]
        # FORM OF CONTACT POINT --> [index normalF shearF1 shearN1x shearN1y shearF2 shearN2x shearN2y]
        # switch x and y because we went into the page to be y not x (different reference frame)
        # here it is assumed that the normal force is always pointed in just z-dir and shear forces have zero z-dir (aka assume table is horizontal)
        particles_force_applied[int(contact_points[j][0])] = [contact_points[j][2]*contact_points[j][4] + contact_points[j][5]*contact_points[j][7] , contact_points[j][2]*contact_points[j][3] + contact_points[j][5]*contact_points[j][6], -1*contact_points[j][1]]
    else:
        particles_force_applied = [0, 0, 0]

@ti.kernel
def initiate_particles():
    unit_dims = dx_dy_dz()
    for i in particles_pos:
        particles_pos[i] = xyz_to_coor(x(i), y(i), z(i))
        particles_vel[i] = [0.0, 0.0, 0.0]
        particles_acc[i] = [0.0, 0.0, 0.0]
        particles_force_applied[i] = [0.0, 0.0, 0.0]
        if(x(i) != 0 and y(i) != 0 and z(i) != 0):
            particles_between[i*3] = [xyz_to_coor(x(i), y(i), z(i))[0] - unit_dims[0]*0.5, xyz_to_coor(x(i), y(i), z(i))[1], xyz_to_coor(x(i), y(i), z(i))[2]]
            particles_between[i*3+1] = [xyz_to_coor(x(i), y(i), z(i))[0], xyz_to_coor(x(i), y(i), z(i))[1] - unit_dims[1]*0.5, xyz_to_coor(x(i), y(i), z(i))[2]]
            particles_between[i*3+2] = [xyz_to_coor(x(i), y(i), z(i))[0], xyz_to_coor(x(i), y(i), z(i))[1], xyz_to_coor(x(i), y(i), z(i))[2] - unit_dims[2]*0.5]
            particles_between_color[i*3] = [1.0, 1.0, 1.0]
            particles_between_color[i*3+1] = [1.0, 1.0, 1.0]
            particles_between_color[i*3+2] = [1.0, 1.0, 1.0]
        if i < extremes_size:
            particles_extremes[i*3] = [0.0, 0.0, 0.0]
            particles_extremes_color[i*3] = [0.0, 0.0, 0.0]
            particles_extremes[i*3+1] = [0.0, 0.0, 0.0]
            particles_extremes_color[i*3+1] = [0.0, 0.0, 0.0]
            particles_extremes[i*3+2] = [0.0, 0.0, 0.0]
            particles_extremes_color[i*3+2] = [0.0, 0.0, 0.0]
            particles_extremes[i*3+3] = [0.0, 0.0, 0.0]
            particles_extremes_color[i*3+3] = [0.0, 0.0, 0.0]

@ti.func
def upper_force(x: ti.i32, y: ti.i32, z: ti.i32):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(particles_pos[index(x,y,z)][0] - particles_pos[index(x,y,z+1)][0])/m
    force_this[1] = k_Spring*(particles_pos[index(x,y,z)][1] - particles_pos[index(x,y,z+1)][1])/m
    force_this[2] = k_Spring*(particles_pos[index(x,y,z)][2] - particles_pos[index(x,y,z+1)][2] + unit_dim[2])/m
    return force_this

@ti.func
def lower_force(x: ti.i32, y: ti.i32, z: ti.i32):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(particles_pos[index(x,y,z)][0] - particles_pos[index(x,y,z-1)][0])/m
    force_this[1] = k_Spring*(particles_pos[index(x,y,z)][1] - particles_pos[index(x,y,z-1)][1])/m
    force_this[2] = k_Spring*(particles_pos[index(x,y,z)][2] - particles_pos[index(x,y,z-1)][2] - unit_dim[2])/m
    return force_this

@ti.func
def lefter_force(x: ti.i32, y: ti.i32, z: ti.i32):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(particles_pos[index(x,y,z)][0] - particles_pos[index(x-1,y,z)][0] - unit_dim[0])/m
    force_this[1] = k_Spring*(particles_pos[index(x,y,z)][1] - particles_pos[index(x-1,y,z)][1])/m
    force_this[2] = k_Spring*(particles_pos[index(x,y,z)][2] - particles_pos[index(x-1,y,z)][2])/m
    return force_this

@ti.func
def righter_force(x: ti.i32, y: ti.i32, z: ti.i32):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(particles_pos[index(x,y,z)][0] - particles_pos[index(x+1,y,z)][0] + unit_dim[0])/m
    force_this[1] = k_Spring*(particles_pos[index(x,y,z)][1] - particles_pos[index(x+1,y,z)][1])/m
    force_this[2] = k_Spring*(particles_pos[index(x,y,z)][2] - particles_pos[index(x+1,y,z)][2])/m
    return force_this

@ti.func
def backer_force(x: ti.i32, y: ti.i32, z: ti.i32):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(particles_pos[index(x,y,z)][0] - particles_pos[index(x,y+1,z)][0])/m
    force_this[1] = k_Spring*(particles_pos[index(x,y,z)][1] - particles_pos[index(x,y+1,z)][1] + unit_dim[1])/m
    force_this[2] = k_Spring*(particles_pos[index(x,y,z)][2] - particles_pos[index(x,y+1,z)][2])/m
    return force_this

@ti.func
def fronter_force(x: ti.i32, y: ti.i32, z: ti.i32):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(particles_pos[index(x,y,z)][0] - particles_pos[index(x,y-1,z)][0])/m
    force_this[1] = k_Spring*(particles_pos[index(x,y,z)][1] - particles_pos[index(x,y-1,z)][1] - unit_dim[1])/m
    force_this[2] = k_Spring*(particles_pos[index(x,y,z)][2] - particles_pos[index(x,y-1,z)][2])/m
    return force_this

@ti.func
def xyz_to_coor(x: ti.i32, y: ti.i32, z: ti.i32):
    output = [x*(width/xNum) - 0.5*width, y*(height/yNum) - 0.5*height, z*(depth/zNum) - 0.5*depth]
    return output

@ti.func
def dx_dy_dz():
    output = [xyz_to_coor(1,0,0)[0] - xyz_to_coor(0,0,0)[0], xyz_to_coor(0,1,0)[1] - xyz_to_coor(0,0,0)[1], xyz_to_coor(0,0,1)[2] - xyz_to_coor(0,0,0)[2]]
    return output

@ti.func
def calc_dist(pos1, pos2):
    return ti.sqrt((pos2[0] - pos1[0])*(pos2[0] - pos1[0]) + (pos2[1] - pos1[1])*(pos2[1] - pos1[1]) + (pos2[2] - pos1[2])*(pos2[2] - pos1[2]))

@ti.func
def calc_force(p1, p2, originalLength):
    unit_dim = dx_dy_dz()
    force_this = ti.Vector([0.0, 0.0, 0.0])    
    force_this[0] = k_Spring*(p2.x - p1.x - unit_dim[0])/m
    force_this[1] = k_Spring*(p2.y - p1.y - unit_dim[1])/m
    force_this[2] = k_Spring*(p2.z - p1.z - unit_dim[2])/m
    return force_this

@ti.func
def calc_color(length: ti.f32, original_length: ti.f32):
    index = length/original_length
    c = ti.Vector([0.0, 0.0, 0.0])
    if(index < 1):
        c = [1.0 - index, 0.0, 0.0]
        return c
    else:
        c = [0.0, 0.0, index]
        return c

@ti.func
def x(i: ti.i32) -> ti.i32:
    x = i % xNum
    return x

@ti.func
def y(i: ti.i32) -> ti.i32:
    y = (i//xNum)%yNum
    return y

@ti.func
def z(i: ti.i32) -> ti.i32:
    z = i // (xNum*yNum)
    return z

@ti.func
def index(x: ti.i32, y: ti.i32, z: ti.i32):
    return (x + y * xNum + z * xNum * yNum)

@ti.kernel
def bulletPos_to_TaichiIndex(x: ti.f32, y: ti.f32, z: ti.f32)-> ti.i32:
    print("bullet position: ["+str(x)+", "+str(y)+", "+str(z)+"]")
    x_new = (x + width*0.5)*(xNum/width)
    y_new = (y + height*0.5)*(yNum/height)
    z_new = (z + depth*0.5)*(zNum/depth)
    pos = [x_new, y_new, z_new]
    print("taichi position: ["+str(x)+", "+str(y)+", "+str(z)+"]")
    return int(index(int(x_new), int(y_new), int(z_new)))


def main():
    t = 0
    index_color = ti.Vector([0.0, 0.0, 0.0])
    initiate_particles()
    window = ti.ui.Window("Taichi FEA Spring Simulation", (800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.5, 0.5, 0.5))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    gui = ti.GUI("FEA SIMULATION")


    while window.running:
        t += 0.01
        ti.sync()
        camera.position(0.0, -30.0, 30)
        camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
        # camera.lookat(0.0, 0.0, 0.0)
        scene.set_camera(camera)
        scene.point_light(pos=(-5, 0, 0), color=(1, 1, 1))
        scene.point_light(pos=(5, 0, 0), color=(1, 1, 1))
        scene.point_light(pos=(0, -20, 10), color=(1, 1, 1))
        # scene.ambient_light((0.5, 0.5, 0.5))

        # scene.particles(particles_pos, radius=0.2, color = (0.0, 0.0, 0.0))
        scene.particles(particles_between, radius = particle_radius, per_vertex_color=particles_between_color)
        scene.particles(cube_center, radius = 1, color = (0.0, 0.0, 0.0))
        #scene.lines(begin = cube_corner_1, end=cube_corner_2, radius=2, color=0x068587)
        canvas.scene(scene)
        window.show()

        # RESET STATE
        if gui.get_event(ti.GUI.PRESS):
            print("press key : {}".format(gui.event.key)) 
            if gui.event.key == 'r':
                #pivot = [0, 0, width]
                #orn = p.getQuaternionFromEuler([0, 0, 0])
                #p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)
                p.resetBasePositionAndOrientation(boxID, [0, 0, depth], [0, 0, 0, 0])
                p.resetBaseVelocity(boxID, [0, 0, 1], [0, 0, 3, 0])
                initiate_particles()


        p.stepSimulation()
        #time.sleep(1./2.)
        contactPoints = p.getContactPoints()
        pos_orient_Bullet = p.getBasePositionAndOrientation(1)
        cube_center[0] = particles_pos[bulletPos_to_TaichiIndex(pos_orient_Bullet[0][0], pos_orient_Bullet[0][1], pos_orient_Bullet[0][2])]
        i = 0
        for points in contactPoints:
            # FORM OF CONTACT POINT --> [index normalF shearF1 shearN1x shearN1y shearF2 shearN2x shearN2y]
            contactPoint_modified = (bulletPos_to_TaichiIndex(contactPoints[i][5][0], contactPoints[i][5][1], contactPoints[i][5][2]), contactPoints[i][9], contactPoints[i][10], contactPoints[i][11][0], contactPoints[i][11][1], contactPoints[i][12], contactPoints[i][13][0], contactPoints[i][13][1])
            contact_points[i] = contactPoint_modified
            i += 1

        
        update_points(t)

        # img = pixels.to_numpy()
        # video_manager.write_frame(img)
# video_manager.make_video(gif=True, mp4=True)

    #TODO: include self-collision handling


if __name__ == '__main__':
    main()

