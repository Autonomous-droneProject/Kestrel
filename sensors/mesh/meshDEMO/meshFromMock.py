import open3d as o3d
import numpy as np
import subprocess
import threading

# Stores most recent sensor readings
latest_points = []

# Function to read from the compiled C++ subprocess
def read_sensor_data():
    global latest_points

    # Start the compiled mock sensor program from mockSensor.cpp
    process = subprocess.Popen(
        ['./sensors/reading/readingDEMO/sensorReading.exe'],  # Change path if needed
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Read line by line from C++ mock data
    while True:
        frame_points = []
        for _ in range(4):  # 4 lines per frame
            line = process.stdout.readline()
            if not line:
                break
            values = list(map(float, line.strip().split(',')))

            # Build x, y, z coordinates (z = depth value)
            row_idx = len(frame_points)
            for col_idx, z in enumerate(values):
                frame_points.append([col_idx, row_idx, z])

        latest_points = frame_points


# Thread to read sensor data continuously
thread = threading.Thread(target=read_sensor_data, daemon=True)
thread.start()

# Visualization loop
vis = o3d.visualization.Visualizer()
vis.create_window()

# Dummy geometry
pcd = o3d.geometry.PointCloud()
mesh = o3d.geometry.TriangleMesh()

while True:
    if not latest_points:
        continue

    # Create Open3D PointCloud object
    np_points = np.array(latest_points)
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # Estimate normals for mesh generation
    pcd.estimate_normals()

    # Ball Pivoting or Poisson surface reconstruction (but simple for 4x4)
    try:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 2.5 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2])
        )
    except:
        continue

    vis.clear_geometries()
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()



