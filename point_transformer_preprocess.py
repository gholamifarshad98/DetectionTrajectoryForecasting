import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.neighbors import KDTree
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_grouping(query_point_index, query_points, grouped_points, all_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the query point and plot it in red
    query_point = query_points[query_point_index]
    ax.scatter(query_point[0], query_point[1], query_point[2], color='r', label='Query Point',s=500)

    # Extract the neighbors' indices for the selected query point
    grouped_point_indices = grouped_points[query_point_index]  # This is a list of indices

    # Debug: Print the indices and corresponding points
    print("Grouped point indices:", grouped_point_indices)
    print("Grouped points coordinates:", all_points[grouped_point_indices])

    # Extract the corresponding coordinates (3D positions) of the neighbors using the indices
    grouped_points_coords = all_points[grouped_point_indices][:, :3]  # Use only x, y, z


    # Plot the neighbors in blue
    ax.scatter(grouped_points_coords[:, 0], grouped_points_coords[:, 1], grouped_points_coords[:, 2], color='b', label='Grouped Neighbors')

    ax.legend()
    plt.show()


# Visualize the point cloud using matplotlib
def plot_point_cloud(points):
    """
    Visualizes the full point cloud in 3D using Matplotlib
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud in 3D
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='b', label='Point Cloud')

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


def query_and_group_with_kdtree(points, k=16):
    """
    Use KD-Tree to find the k-nearest neighbors of each point in the point cloud
    """
    # Build a KD-Tree from the point cloud
    kdtree = KDTree(points)

    # Query the k-nearest neighbors for each point
    neighbors = []
    for point in points:
        # Find the k nearest neighbors of the current point
        dist, ind = kdtree.query([point], k=k)  # Returns indices and distances of the k nearest neighbors
        neighbors.append(ind[0])

    return np.array(neighbors)

def query_and_group_with_radius(points, radius):
    """
    Use KD-Tree to find all points within a specified radius of each point in the point cloud.
    """
    # Build a KD-Tree from the point cloud
    kdtree = KDTree(points)

    # Query all points within the specified radius for each point
    neighbors = []
    for point in points:
        # Find all points within the radius of the current point
        ind = kdtree.query_radius([point], r=radius)  # Returns indices of points within the radius
        neighbors.append(ind[0])  # Append the indices of the neighbors

    return np.array(neighbors, dtype=object)  # Use dtype=object because the number of neighbors varies


def furthest_point_sampling(p, npoints):
    """
    Selects npoints furthest points from the set of points p.
    Args:
        p: (n_points, 3) point cloud.
        npoints: Number of points to sample.
    Returns:
        centroids: (npoints, 3) selected farthest points.
    """
    # Initialize with a random point
    idx = torch.randint(0, p.size(0), (1,)).long()
    centroids = p[idx]
    
    dist = torch.norm(p - centroids, p=2, dim=1)  # Compute the distance from the first point

    for i in range(1, npoints):
        dist = torch.min(dist, torch.norm(p - centroids[-1], p=2, dim=1))  # Update distances
        farthest_idx = torch.argmax(dist)  # Get the farthest point
        centroids = torch.cat((centroids, p[farthest_idx].unsqueeze(0)), dim=0)
    
    return centroids


def sum_features(features, nsample):
    """
    Sum features over the nearest neighbors.
    Args:
        features: (n_query_points, nsample, c) features of the neighbors.
        nsample: Number of neighbors.
    """
    return features.sum(dim=1)  # Sum along the nsample axis: (n_query_points, c)


# Initialize NuScenes object
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes/v1.0-mini', verbose=True)

# Example: Load the first sample data from the dataset
scene = nusc.scene[3]
sample = nusc.get('sample', scene['first_sample_token'])

# Get the Lidar data (assuming lidar data is present)
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

# Load the point cloud from file
pc = LidarPointCloud.from_file(lidar_path)
points = torch.tensor(pc.points.T, dtype=torch.float32)  # (N, 3)

# Optionally, get features (if available) or use a dummy feature tensor
features = torch.ones_like(points[:3])  # Dummy feature vector

# Define the number of nearest neighbors
nsample = 1000

# Example: query points from the first 100 points
p_query = points[:1000]  # Query the first 100 points

# Call the functions for grouping neighbors and farthest point sampling

# grouped_neighbors = query_and_group_with_kdtree(points.numpy(), k=nsample)  # KD-Tree grouping
grouped_neighbors = query_and_group_with_radius(points.numpy(), 10)  # KD-Tree grouping with radius
farthest_points = furthest_point_sampling(points, npoints=1000)

# Display the results
print("Grouped neighbors (indices):", grouped_neighbors.shape)
print("Farthest points:", farthest_points.shape)

# Visualize the first 1000 points (to avoid cluttering the plot)
plot_point_cloud(points[:].numpy())

# Visualize the grouping for a specific query point (e.g., index 0)
query_point_index = 0  # Choose a query point index (can be any integer in range of number of query points)
visualize_grouping(query_point_index, p_query.numpy(), grouped_neighbors, points.numpy())
