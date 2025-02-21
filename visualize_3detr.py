import open3d as o3d
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.optimize import linear_sum_assignment





# ----------------------
# 1. Backbone Network (PointNet++ Style)
# ----------------------
class FPSDownsample(nn.Module):
    """Farthest Point Sampling with MLP feature extraction"""
    def __init__(self, in_dim=3, out_dim=256, n_samples=1024):
        super().__init__()
        self.n_samples = n_samples
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def farthest_point_sample(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        centroids = torch.zeros(B, self.n_samples, dtype=torch.long, device=x.device)
        distance = torch.ones(B, N, device=x.device) * 1e10
        
        # Randomly select first point
        centroid = torch.randint(0, N, (B,), dtype=torch.long, device=x.device)
        
        for i in range(self.n_samples):
            centroids[:, i] = centroid
            centroid_points = x[torch.arange(B), centroid, :]  # (B, 3)
            dist = torch.sum((x - centroid_points.unsqueeze(1)) ** 2, -1)  # (B, N)
            mask = dist < distance
            distance[mask] = dist[mask]
            centroid = torch.argmax(distance, -1)  # (B,)
        return centroids

    def forward(self, x):
        # x: (B, N, 3)
        B = x.shape[0]
        
        # FPS Sampling
        fps_idx = self.farthest_point_sample(x)  # (B, n_samples)
        sampled_points = x[torch.arange(B).unsqueeze(-1), fps_idx]  # (B, n_samples, 3)
        
        # Extract features
        features = self.mlp(sampled_points)  # (B, n_samples, out_dim)
        return sampled_points, features

# ----------------------
# 2. Transformer Architecture
# ----------------------
class Transformer3DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100):
        super().__init__()
        # Backbone
        self.backbone = FPSDownsample(out_dim=hidden_dim, n_samples=1024)
        
        # Transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Learnable object queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)  # (x, y, z, w, l, h, rot)
        )
    def forward(self, x):
        # x: (B, N, 3)
        
        # 1. Backbone processing
        _, features = self.backbone(x)  # (B, 1024, hidden_dim)
        
        # 2. Encoder
        encoded = self.encoder(features)  # (B, 1024, hidden_dim)
        
        # 3. Decoder (expand queries to batch size)
        queries = self.queries.unsqueeze(0).repeat(x.size(0), 1, 1)  # (B, num_queries, hidden_dim)
        decoded = self.decoder(queries, encoded)  # (B, num_queries, hidden_dim)
        
        # 4. Predictions
        class_logits = self.class_head(decoded)  # (B, num_queries, num_classes)
        bbox_preds = self.bbox_head(decoded)     # (B, num_queries, 7)
        
        return {'pred_logits': class_logits, 'pred_boxes': bbox_preds}

# ----------------------
# 3. Hungarian Loss
# ----------------------
class HungarianLoss(nn.Module):
    def __init__(self, num_classes=7, class_weight=1.0, box_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.box_weight = box_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys:
                'pred_logits': (batch_size, num_queries, num_classes)
                'pred_boxes': (batch_size, num_queries, 7)
            targets: list of dicts (length=batch_size) with keys:
                'labels': (num_gt_boxes,)
                'boxes': (num_gt_boxes, 7)
        """
        batch_size = outputs['pred_logits'].shape[0]
        total_loss = 0
        
        for batch_idx in range(batch_size):
            # Get predictions for this sample
            pred_logits = outputs['pred_logits'][batch_idx]  # (100, 7)
            pred_boxes = outputs['pred_boxes'][batch_idx]    # (100, 7)
            
            # Get ground truth for this sample
            gt_labels = targets[batch_idx]['labels']  # (num_gt,)
            gt_boxes = targets[batch_idx]['boxes']     # (num_gt, 7)
            num_gt = gt_labels.shape[0]
            
            # Create cost matrix (100 x num_gt)
            cost_matrix = self._compute_cost_matrix(
                pred_logits, pred_boxes,
                gt_labels, gt_boxes
            )
            
            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            
            # Compute losses
            loss = self._compute_loss(
                pred_logits, pred_boxes,
                gt_labels, gt_boxes,
                row_ind, col_ind
            )
            
            total_loss += loss
            
        return total_loss / batch_size

    def _compute_cost_matrix(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        # Classification cost
        log_probs = F.log_softmax(pred_logits, dim=-1)  # (100, 7)
        class_cost = -log_probs[:, gt_labels]           # (100, num_gt)
        
        # Box regression cost
        box_cost = torch.cdist(pred_boxes[:, :3], gt_boxes[:, :3], p=1)  # (100, num_gt)
        
        return self.class_weight * class_cost + self.box_weight * box_cost

    def _compute_loss(self, pred_logits, pred_boxes, gt_labels, gt_boxes, row_ind, col_ind):
        # Classification loss
        class_loss = F.cross_entropy(
            pred_logits[row_ind],  # (num_matched, 7)
            gt_labels[col_ind]     # (num_matched,)
        )
        
        # Box regression loss
        box_loss = F.l1_loss(
            pred_boxes[row_ind],   # (num_matched, 7)
            gt_boxes[col_ind]      # (num_matched, 7)
        )
        
        return self.class_weight * class_loss + self.box_weight * box_loss

# ----------------------
# 4. Data Loading Utilities
# ----------------------
def load_lidar_data(nusc, sample_token, max_points=35000):
    # Load and downsample LiDAR data
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_data['token']))
    
    # Convert to tensor
    points = torch.tensor(pc.points[:3].T, dtype=torch.float32)
    
    # Random subsampling if needed
    if points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0])[:max_points]
        points = points[idx]
    
    return points.unsqueeze(0)  # Add batch dimension

# Define class names and colors
CLASS_NAMES = ['human.pedestrian.adult', 'vehicle.car', 'vehicle.truck', 'vehicle.motorcycle', 'vehicle.bicycle', 'movable_object.barrier', 'vehicle.construction']
CLASS_COLORS = {
    'human.pedestrian.adult': [0.0, 0.5, 1.0],
    'vehicle.car': [1.0, 0.0, 0.0],
    'vehicle.truck': [0.0, 1.0, 0.0],
    'vehicle.motorcycle': [0.5, 0.5, 0.5],
    'vehicle.bicycle': [1.0, 0.5, 0.0],
    'movable_object.barrier': [0.0, 1.0, 1.0],
    'vehicle.construction': [0.0, 0.0, 1.0]
}

def compute_box_corners(box):
    """Compute the 8 corners of a bounding box."""
    x, y, z, w, l, h, yaw = box[:7]
    corners = np.array([
        [ w/2,  l/2, -h/2], [ w/2, -l/2, -h/2], [-w/2, -l/2, -h/2], [-w/2,  l/2, -h/2],
        [ w/2,  l/2,  h/2], [ w/2, -l/2,  h/2], [-w/2, -l/2,  h/2], [-w/2,  l/2,  h/2]
    ])
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    corners = np.dot(corners, rotation_matrix.T) + np.array([x, y, z])
    return corners

def visualize_predictions(data_list, score_threshold):
    """Visualize all frames in a single Open3D window with bigger labels."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for frame_data in data_list:
        vis.clear_geometries()

        # Add point cloud
        points = frame_data['points']
        pcd = o3d.geometry.PointCloud()

        non_ground_mask = points[:, 2] > -1.5  # Keep points above the threshold
        points = points[non_ground_mask]

        # Ensure points are compatible with Open3D
        points = np.asarray(points[:, :3], dtype=np.float32)  # Extract x, y, z and ensure correct type
        if points.ndim != 2 or points.shape[1] != 3:  # Check shape after extraction
            raise ValueError(f"Invalid points shape after extraction: {points.shape}. Expected (N, 3).")

        pcd.points = o3d.utility.Vector3dVector(points)
        custom_color = np.array([0.0, 0.0, 0.0])  # RGB values for light blue
        colors = np.tile(custom_color, (points.shape[0], 1))  # Apply the color to all points
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)

        # Add bounding boxes and labels
        for i, box in enumerate(frame_data['boxes']):
            if frame_data['scores'][i] < score_threshold:
                continue

            corners = compute_box_corners(box)
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # Set box color based on class
            label = frame_data['labels'][i]
            class_name = CLASS_NAMES[label]
            color = CLASS_COLORS.get(class_name, [1.0, 1.0, 1.0])
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
            vis.add_geometry(line_set)

            # Add larger sphere at the center as label placeholder
            center = np.mean(corners, axis=0)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # Bigger radius for visibility
            sphere.translate(center)
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere)

            # Display label text in the console
            score = frame_data['scores'][i]
            print(f"Label: {class_name} | Score: {score:.2f} | Center: {center}")

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)  # Adjust playback speed

    vis.destroy_window()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='Visualize predictions in a single Open3D window')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--score_threshold', type=float, default=0.3, help='Score threshold for filtering predictions')
    args = parser.parse_args()

    # Initialize NuScenes dataset
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/farshad/Desktop/AI/OpenPCDet/data/nuscenes/v1.0-mini', verbose=False)

    # Load the trained model
    model = Transformer3DETR(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    # Preprocess frames and store data
    data_list = []
    with torch.no_grad():
        for sample in nusc.sample:
            sample_token = sample['token']
            
            # Load LiDAR data
            lidar = load_lidar_data(nusc, sample_token).to(device)
            
            # Forward pass
            outputs = model(lidar)
            
            # Extract predictions
            pred_boxes = outputs['pred_boxes'].cpu().numpy()
            pred_scores = outputs['pred_logits'].softmax(dim=-1).max(dim=-1).values.squeeze().cpu().numpy()  # Squeeze to remove batch dimension
            pred_labels = outputs['pred_logits'].argmax(dim=-1).squeeze().cpu().numpy()  # Squeeze to remove batch dimension
            
            data_list.append({
                'points': lidar.squeeze(0).cpu().numpy(),
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels,
            })

    # Visualize all frames
    visualize_predictions(data_list, args.score_threshold)

if __name__ == '__main__':
    main()
