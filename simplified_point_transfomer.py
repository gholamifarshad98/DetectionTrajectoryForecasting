import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.neighbors import KDTree
import os
import torch.nn as nn
import torch.nn.functional as F

# Define the Point Transformer Layer
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, nsample=16):
        super(PointTransformerLayer, self).__init__()
        self.nsample = nsample
        
        # Linear layers for query, key, value, and position encoding
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        
        # Position encoding
        self.linear_p = nn.Sequential(
            nn.Linear(3, out_planes),  # Ensure output size matches the feature size
            nn.BatchNorm1d(out_planes),  # Ensure BatchNorm1d is matching out_planes
            nn.ReLU(inplace=True)
        )
        
        # MLP for attention weights
        self.mlp_gamma = nn.Sequential(
            nn.Linear(out_planes, out_planes),
            nn.BatchNorm1d(out_planes),  # Ensure BatchNorm1d is matching out_planes
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes)
        )
        
        # Final linear layer
        self.linear_out = nn.Linear(out_planes, out_planes)
        
    def forward(self, points, features):
        """
        Args:
            points: (N, 3) tensor of point coordinates
            features: (N, C) tensor of point features
        Returns:
            transformed_features: (N, out_planes) tensor of transformed features
        """
        N = points.shape[0]
        
        # Query, Key, Value transformations
        q = self.linear_q(features)  # (N, out_planes)
        k = self.linear_k(features)  # (N, out_planes)
        v = self.linear_v(features)  # (N, out_planes)

        # Grouping: Find nearest neighbors for each point
        grouped_indices = query_and_group_with_kdtree(points.numpy(), k=self.nsample)  # (N, nsample)
        grouped_indices = torch.tensor(grouped_indices, dtype=torch.long)  # Convert to tensor

        # Gather the grouped points and features
        grouped_points = points[grouped_indices]  # (N, nsample, 3)
        grouped_k = k[grouped_indices]  # (N, nsample, out_planes)
        grouped_v = v[grouped_indices]  # (N, nsample, out_planes)

        # Position encoding: Calculate the relative positions of neighbors
        relative_pos = grouped_points - points.unsqueeze(1)  # (N, nsample, 3)

        # Flatten relative positions to pass them through the linear layer
        p = relative_pos.view(-1, 3)  # (N * nsample, 3)
        p = self.linear_p(p)  # (N * nsample, out_planes)

        # Reshape p to (N, nsample, out_planes) to match q_expanded and grouped_k
        p = p.view(N, self.nsample, -1)  # (N, nsample, out_planes)

        # Ensure q has the shape (N, nsample, out_planes) by expanding it
        q_expanded = q.unsqueeze(1).expand(-1, self.nsample, -1)  # (N, nsample, out_planes)

        # Now q_expanded, grouped_k, and p can be added/subtracted
        attention_input = q_expanded - grouped_k + p  # (N, nsample, out_planes)

        # Reshape attention_input to (N * nsample, out_planes) for mlp_gamma
        attention_input = attention_input.view(-1, self.linear_out.out_features)  # (N * nsample, out_planes)
        attention_scores = self.mlp_gamma(attention_input)  # (N * nsample, out_planes)

        # Reshape attention_scores back to (N, nsample, out_planes)
        attention_scores = attention_scores.view(N, self.nsample, -1)  # (N, nsample, out_planes)

        # Apply softmax to get attention weights
        attention_scores = F.softmax(attention_scores, dim=1)  # (N, nsample, out_planes)

        # Weighted sum of values
        transformed_features = torch.sum((grouped_v + p) * attention_scores, dim=1)  # (N, out_planes)
        
        # Final transformation
        transformed_features = self.linear_out(transformed_features)
        
        return transformed_features
# Define the Point Transformer Network
class PointTransformer(nn.Module):
    def __init__(self, in_planes, out_planes, num_layers=4, nsample=16):
        super(PointTransformer, self).__init__()
        self.layers = nn.ModuleList([
            PointTransformerLayer(in_planes if i == 0 else out_planes, out_planes, nsample)
            for i in range(num_layers)
        ])
        
    def forward(self, points, features):
        for layer in self.layers:
            features = layer(points, features)
        return features


# Function to group points using KD-Tree
def query_and_group_with_kdtree(points, k=16):
    """
    Use KD-Tree to find the k-nearest neighbors of each point in the point cloud
    """
    # Only use the first 3 dimensions (x, y, z) for KD-Tree
    points_3d = points[:, :3]  # (N, 3)
    kdtree = KDTree(points_3d)

    neighbors = []
    for point in points_3d:
        dist, ind = kdtree.query([point], k=k)  # Returns indices and distances of the k nearest neighbors
        neighbors.append(ind[0])

    return np.array(neighbors)


# Main script
# Main script
if __name__ == "__main__":
    # Initialize NuScenes object
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/sohail/AI/v1.0-mini/nuscenes', verbose=True)

    # Example: Load the first sample data from the dataset
    scene = nusc.scene[3]
    sample = nusc.get('sample', scene['first_sample_token'])

    # Get the Lidar data (assuming lidar data is present)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    # Load the point cloud from file
    pc = LidarPointCloud.from_file(lidar_path)
    # points = torch.tensor(pc.points.T, dtype=torch.float32)  # (N, 3)
    points = torch.tensor(pc.points[:3, :].T, dtype=torch.float32)  # Only use the first 3 dimensions (x, y, z)


    # Normalize point coordinates (optional)
    points -= points.mean(dim=0)  # Center the point cloud
    points /= points.std(dim=0)   # Normalize scale

    # Create dummy features (if no features are available)
    features = torch.ones_like(points[:, :3])  # (N, 3)

    # Initialize the Point Transformer
    model = PointTransformer(in_planes=3, out_planes=64, num_layers=4, nsample=16)

    # Forward pass: Transform the features
    transformed_features = model(points, features)
    print("Transformed features shape:", transformed_features.shape)