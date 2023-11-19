import numpy as np
import open3d as o3d    # for visualization

output_directory = "./output/"

def rad(deg):
    return (deg * np.pi) / 180.0

SPEED = 10
LINES_PER_SEC = 10

ANGULAR_STEP_WIDTH = rad(1.8)
ANGULAR_UNCERTAINTY = rad(0.1)
TRAJECTORY_UNCERTAINTY = rad(2.0)

LINE_SPACE = 10.0
AGL = 40.0
RANGE = 80.0
DOMAIN = [40.0, 100.0]

# Surface Representation -- Signed Distance Field (SDF)

def ground_sdf(p):
    return (p[2])

def box_sdf(p, c, b):
  q = np.abs(p - c) - b
  return np.linalg.norm(np.maximum(q, 0.0)) + np.minimum(np.max(q), 0.0)

def sphere_sdf(p, c, R):
    return np.linalg.norm(p - c) - R

def cubes_sdf(p):
    distance = RANGE
    for i in np.linspace(0, DOMAIN[0], 2):
        for j in np.linspace(0, DOMAIN[1], 5):
            distance = min(distance, box_sdf(p, np.array([i, j, 2.0]), np.array([4.0, 4.0, 4.0])))
    return distance

def sdf(p):
    # a hemisphere placed on the ground
    return min(cubes_sdf(p), ground_sdf(p))

# LiDAR Data Collection

def sph_vec(phi, theta, r=1.0):
    return r * np.array([ np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) ])

def cast_line(r, u, epsilon=0.1, max=RANGE):
    # cast line in direction u from r until intersection; return intersection
    p = np.array(r)
    distance = sdf(p)
    while (distance > epsilon):
        if (np.linalg.norm(p - r) >= max):
            return np.zeros(3)
        p += distance * u
        distance = sdf(p)
    return p

def noise(size):
    return size * np.random.rand(1)[0]

def take_line_snapshot(p, phi, angular_step=ANGULAR_STEP_WIDTH, epsilon=0.1):
    # takes a snapshot along a specific slice / line
    points = []
    for theta in np.arange(np.pi / 2, (3 * np.pi / 2) + angular_step, angular_step):
        point = cast_line(p, sph_vec(phi + noise(TRAJECTORY_UNCERTAINTY), theta + noise(ANGULAR_UNCERTAINTY)))
        if (np.linalg.norm(point) > epsilon):
            # print(f"raycast @ ({phi}, {theta}) ==> {point}")
            points.append(point)
    print(f"line snapshot @ {p}, phi = {(phi * 180) / np.pi}")
    return np.asarray(points)

# def take_360_snapshot(p, lines_per_scan, epsilon=0.1):
#     # sample evenly distributed directions (360 deg fov) 
#     # (can be parallelized for high performance)
#     points = []
#     for theta in np.linspace(0, np.pi, int((lines_per_scan) / 3)):
#         for phi in np.linspace(0, 2*np.pi, int((2 * lines_per_scan) / 3)):
#             point = cast_line(p, sph_vec(phi + noise(0.05), theta + noise(0.05)))
#             if (np.linalg.norm(point) > epsilon):
#                 # print(f"raycast @ ({phi}, {theta}) ==> {point}")
#                 points.append(point)
#     print(f"snapshot @ {p}")
#     return np.asarray(points)

# LiDAR Flight Path

def flight_positions(step=10.0):
    positions = []
    for i in np.arange(0.0, DOMAIN[0] + LINE_SPACE, LINE_SPACE):
        line = []
        for j in np.arange(0.0, DOMAIN[1] + step, step):
            line.append(np.array([i + noise(1.0), j + noise(1.0), AGL + noise(1.0)]))
        positions.append(line)
    return positions

# LiDAR Simulation

def simulate_lidar(filename="scan"):
    lines = flight_positions(SPEED / LINES_PER_SEC)
    snapshots = []
    for line in lines:
        for p in line:
            snapshots.extend(take_line_snapshot(p, 0))
    export_points(snapshots, filename)

# Exporting as a Point Cloud

def export_points(points, filename="scan"):
    print(f"exporting {points} points")
    colors = np.array([ [1, 0, 0] for point in points ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename=output_directory + filename + ".pcd", pointcloud=pcd)
    o3d.visualization.draw_geometries([pcd])

def view_simulated_scan(filename="scan"):
    pcd = o3d.io.read_point_cloud(output_directory + filename + ".pcd")
    print(len(pcd.points), "points")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # simulate_lidar("lines")
    view_simulated_scan("lines")