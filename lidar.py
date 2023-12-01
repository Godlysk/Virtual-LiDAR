import numpy as np
import pandas as pd
import open3d as o3d    # for visualization

output_directory = "./output/"

def to_rad(deg):
    return (deg * np.pi) / 180.0


SPEED = 1
LINES_PER_SEC = 1

ANGULAR_STEP_WIDTH = to_rad(1.8)
ANGULAR_UNCERTAINTY = to_rad(0.1)
TRAJECTORY_UNCERTAINTY = to_rad(2.0)
BEAM_DIVERGENCE = 0.0003
WAIST_SCALE = 2.0  # exaggerate effects of beam divergence (?)

LINE_SPACE = 2.5
AGL = 10.0
RANGE = 10.0
DOMAIN = [5.0, 5.0]

# Surface Representation -- Signed Distance Field (SDF)

def ground_sdf(p):
    return (p[2])

def get_tree_representation():
    tree_df = pd.read_csv("tree.csv", sep=',', header=None)
    tree = tree_df.values
    points = 1154
    vertices = tree[0].reshape(points, 3)
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    indices = tree[1, :points * 2].reshape(points, 2)
    return (vertices, indices)

def segment_sdf(p, a, b, r):
    pa = p - a
    ba = b - a
    h = np.clip(np.dot(pa,ba)/np.dot(ba,ba), 0.0, 1.0)
    return np.linalg.norm(pa - ba*h) - r

def tree_sdf(p, vertices, indices):
    d = RANGE
    for pair in indices:
        a, b = vertices[int(pair[0])], vertices[int(pair[1])]
        d = min(segment_sdf(p, a, b, 0.05), d)
    return d

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

def sdf(p, sdf_data):
    # a hemisphere placed on the ground
    # return min(cubes_sdf(p), ground_sdf(p))
    return min(tree_sdf(p, sdf_data[0], sdf_data[1]), ground_sdf(p))

# LiDAR Beam (Gaussian or Ray)

def waist(h):
    # beam waist at 1/e^2 (see Gaussian Beam) -- empirical model
    # todo: replace magic numbers with constant w_0
    return 8.1 * np.sqrt(1 + (h / 27.0) ** 2) * 0.001 * WAIST_SCALE

def intensity(h, d): # r = d * waist(h)
    return (1 + (h / 27.0) ** 2) * np.exp(-2 * d * d)

def orth_vec(dir, theta):
    orth = np.array([dir[2], 0, -dir[0]])
    orthorth = np.cross(orth, dir)
    orth_hat = orth / np.linalg.norm(orth)
    orthorth_hat = orthorth / np.linalg.norm(orth)
    return np.cos(theta) * orth_hat + np.sin(theta) * orthorth_hat

def compress_returns(return_heights, epsilon=1.0):
    # it could be interesting making a histogram plot with all of these returns
    # todo: find a way to factor the intensity into the compressed returns
    sorted = np.sort(return_heights)
    # print(sorted) # print out the sorted returns (debug tool)
    compressed = [ sorted[0] ]
    for i in range(1, len(sorted)):
        if (sorted[i] - sorted[i - 1] > epsilon):
            compressed.append(sorted[i])
    return compressed # we only add a return to compressed if it is sufficiently far from other returns

def sph_vec(phi, theta, r=1.0):
    return r * np.array([ np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) ])

def cast_line(r, u, sdf_data, epsilon=0.1, max=RANGE):
    # cast line in direction u from r until intersection; return intersection
    p = np.array(r)
    distance = sdf(p, sdf_data)
    while (distance > epsilon):
        if (np.linalg.norm(p - r) >= max):
            return np.zeros(3)
        p += distance * u
        distance = sdf(p, sdf_data)
    return p

def cast_gaussian_line(r, u, theta, d, sdf_data, epsilon=0.1, max=RANGE):
    # cast gaussian line in direction u from r with angle theta and centre-offset d (as a ratio of w_0)
    h = 0
    orth = orth_vec(u, theta) * d
    p = np.array(r) + orth * waist(0)
    distance = sdf(p, sdf_data)
    while (distance > epsilon and h < max):
        h += distance
        distance = sdf(waist(h) * orth + np.array(r) + u * h, sdf_data)
    return h

def cast_gaussian_beam(r, u, sdf_data, epsilon=0.1, max=RANGE, beams=10):
    returns = []
    for i in range(beams):
        d, angle = np.random.rand(2)
        h = cast_gaussian_line(r, u, angle*2*np.pi, d, sdf_data, epsilon, max)
        # todo: find a way to factor the intensity into the compressed returns
        val = intensity(h, d)
        returns.append(h)
    return compress_returns(returns)

# LiDAR Data Collection

def noise(size):
    return size * np.random.rand(1)[0]

def take_line_snapshot(p, phi, sdf_data, angular_step=ANGULAR_STEP_WIDTH, epsilon=0.1):
    # takes a snapshot along a specific slice / line
    points = []
    for theta in np.arange(np.pi / 2, (3 * np.pi / 2) + angular_step, angular_step):
        # replace with beam divergence casting and store multiple returns associated with the same point
        # replace point with points for multiple returns
        point = cast_line(p, sph_vec(phi + noise(TRAJECTORY_UNCERTAINTY), theta + noise(ANGULAR_UNCERTAINTY)), sdf_data)
        if (np.linalg.norm(point) > epsilon):
            # print(f"raycast @ ({phi}, {theta}) ==> {point}")
            points.append(point)
    print(f"line snapshot @ {p}, phi = {(phi * 180) / np.pi}")
    return np.asarray(points)


def take_gaussian_line_snapshot(p, phi, sdf_data, angular_step=ANGULAR_STEP_WIDTH, max=RANGE):
    # takes a snapshot along a slice using gaussian beams (there may be multiple returns for each beam)
    points = []
    for theta in np.arange(np.pi / 2, (3 * np.pi / 2) + angular_step, angular_step):
        u = sph_vec(phi + noise(TRAJECTORY_UNCERTAINTY), theta + noise(ANGULAR_UNCERTAINTY))
        heights = cast_gaussian_beam(p, u, sdf_data)
        for height in heights:
            if (height < max):
                # print(f"raycast @ ({phi}, {theta}) ==> {point}")
                points.append(height * u + p)
    print(f"\nline snapshot @ {p}, phi = {(phi * 180) / np.pi}")
    return np.asarray(points)

# LiDAR Flight Path

def flight_positions(step=10.0):
    positions = []
    # choose if negative regions are included or not
    for i in np.arange(-DOMAIN[0], DOMAIN[0] + LINE_SPACE, LINE_SPACE):
        line = []
        for j in np.arange(-DOMAIN[1], DOMAIN[1] + step, step):
            line.append(np.array([i + noise(1.0), j + noise(1.0), AGL + noise(1.0)]))
        positions.append(line)
    return positions

# LiDAR Simulation

def simulate_lidar(sdf_data, filename="scan"):
    lines = flight_positions(SPEED / LINES_PER_SEC)
    snapshots = []
    for line in lines:
        for p in line:
            # gaussian or normal beams
            # snapshots.extend(take_line_snapshot(p, 0, sdf_data))
            snapshots.extend(take_gaussian_line_snapshot(p, 0, sdf_data))
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
    simulate_lidar( get_tree_representation(), "gaussian")
    # view_simulated_scan("lines")