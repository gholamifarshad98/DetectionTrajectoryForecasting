import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import math

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(self, d_pos=256, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        assert d_pos % 6 == 0, "d_pos must be divisible by 6 for 3D sine embedding"
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, xyz, input_range=None):
        if self.normalize:
            assert input_range is not None, "input_range must be provided for normalization"
            min_coords, max_coords = input_range
            xyz = (xyz - min_coords) / (max_coords - min_coords) * self.scale

        dim_t = torch.arange(self.d_pos // 6, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_pos // 6))

        pos_x = xyz[:, :, 0:1] / dim_t
        pos_y = xyz[:, :, 1:2] / dim_t
        pos_z = xyz[:, :, 2:3] / dim_t

        pos_x = torch.cat((pos_x.sin(), pos_x.cos()), dim=2)
        pos_y = torch.cat((pos_y.sin(), pos_y.cos()), dim=2)
        pos_z = torch.cat((pos_z.sin(), pos_z.cos()), dim=2)

        pos_embed = torch.cat((pos_x, pos_y, pos_z), dim=2)
        return pos_embed

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

class Transformer3DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, nhead=8):
        super().__init__()
        if hidden_dim % nhead != 0:
            hidden_dim = (hidden_dim // nhead) * nhead
            print(f"Adjusted hidden_dim to {hidden_dim} to be divisible by nhead={nhead}")

        # Backbone
        self.backbone = FPSDownsample(out_dim=hidden_dim, n_samples=1024)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=4
        )

        # Learnable object queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Positional embedding
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=hidden_dim)

        # Transformer Decoder with intermediate outputs
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=4
        )

        # Detailed MLP Heads for Box Predictions
        self.class_head = nn.Linear(hidden_dim, num_classes)  # Class prediction head
        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Predicts (x, y, z) center offsets
        )
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Predicts (w, l, h) dimensions
        )
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predicts yaw angle (rotation around z-axis)
        )

    def forward(self, x):
        # x: (B, N, 3)

        # 1. Backbone processing
        sampled_points, features = self.backbone(x)  # (B, 1024, hidden_dim)

        # 2. Encoder
        encoded = self.encoder(features)  # (B, 1024, hidden_dim)

        # 3. Positional embedding for encoder output
        input_range = (x.min(dim=1)[0], x.max(dim=1)[0])  # (B, 3), (B, 3)
        pos_embed = self.pos_embedding(sampled_points, input_range)  # (B, 1024, hidden_dim)

        # Add positional embedding to encoder output
        encoded = encoded + pos_embed

        # 4. Decoder (expand queries to batch size)
        queries = self.queries.unsqueeze(0).repeat(x.size(0), 1, 1)  # (B, num_queries, hidden_dim)

        # Decode with intermediate outputs
        intermediate_outputs = []
        for layer in self.decoder.layers:
            queries = layer(queries, encoded)
            intermediate_outputs.append(queries)

        # Stack intermediate outputs
        intermediate_outputs = torch.stack(intermediate_outputs)  # (num_layers, B, num_queries, hidden_dim)

        # 5. Predictions for each intermediate output
        all_class_logits = []
        all_center_preds = []
        all_size_preds = []
        all_angle_preds = []
        for output in intermediate_outputs:
            class_logits = self.class_head(output)  # (B, num_queries, num_classes)
            center_preds = self.center_head(output)  # (B, num_queries, 3)
            size_preds = self.size_head(output)      # (B, num_queries, 3)
            angle_preds = self.angle_head(output)    # (B, num_queries, 1)
            all_class_logits.append(class_logits)
            all_center_preds.append(center_preds)
            all_size_preds.append(size_preds)
            all_angle_preds.append(angle_preds)

        # Stack predictions
        all_class_logits = torch.stack(all_class_logits)  # (num_layers, B, num_queries, num_classes)
        all_center_preds = torch.stack(all_center_preds)  # (num_layers, B, num_queries, 3)
        all_size_preds = torch.stack(all_size_preds)      # (num_layers, B, num_queries, 3)
        all_angle_preds = torch.stack(all_angle_preds)    # (num_layers, B, num_queries, 1)

        return {
            'pred_logits': all_class_logits[-1],  # Final class predictions
            'pred_boxes': {
                'center': all_center_preds[-1],  # Final center predictions
                'size': all_size_preds[-1],     # Final size predictions
                'angle': all_angle_preds[-1],   # Final angle predictions
            },
            'aux_outputs': {
                'pred_logits': all_class_logits[:-1],  # Intermediate class predictions
                'pred_boxes': {
                    'center': all_center_preds[:-1],  # Intermediate center predictions
                    'size': all_size_preds[:-1],     # Intermediate size predictions
                    'angle': all_angle_preds[:-1],   # Intermediate angle predictions
                }
            }
        }

# Updated Hungarian Loss for Detailed MLP Heads
class HungarianLoss(nn.Module):
    def __init__(self, num_classes=7, class_weight=1.0, center_weight=1.0, size_weight=1.0, angle_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.angle_weight = angle_weight

    def forward(self, outputs, targets):
        batch_size = outputs['pred_logits'].shape[0]
        total_loss = 0
        
        for batch_idx in range(batch_size):
            # Get predictions for this sample
            pred_logits = outputs['pred_logits'][batch_idx]  # (num_queries, num_classes)
            pred_center = outputs['pred_boxes']['center'][batch_idx]  # (num_queries, 3)
            pred_size = outputs['pred_boxes']['size'][batch_idx]      # (num_queries, 3)
            pred_angle = outputs['pred_boxes']['angle'][batch_idx]    # (num_queries, 1)
            
            # Get ground truth for this sample
            gt_labels = targets[batch_idx]['labels']  # (num_gt,)
            gt_boxes = targets[batch_idx]['boxes']     # (num_gt, 7)
            num_gt = gt_labels.shape[0]
            
            # Create cost matrix (num_queries x num_gt)
            cost_matrix = self._compute_cost_matrix(
                pred_logits, pred_center, pred_size, pred_angle,
                gt_labels, gt_boxes
            )
            
            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            
            # Compute losses
            loss = self._compute_loss(
                pred_logits, pred_center, pred_size, pred_angle,
                gt_labels, gt_boxes,
                row_ind, col_ind
            )
            
            total_loss += loss
            
        return total_loss / batch_size

    def _compute_cost_matrix(self, pred_logits, pred_center, pred_size, pred_angle, gt_labels, gt_boxes):
        # Classification cost
        log_probs = F.log_softmax(pred_logits, dim=-1)  # (num_queries, num_classes)
        class_cost = -log_probs[:, gt_labels]           # (num_queries, num_gt)
        
        # Center regression cost
        center_cost = torch.cdist(pred_center, gt_boxes[:, :3], p=1)  # (num_queries, num_gt)
        
        # Size regression cost
        size_cost = torch.cdist(pred_size, gt_boxes[:, 3:6], p=1)     # (num_queries, num_gt)
        
        # Angle regression cost
        angle_cost = torch.abs(pred_angle - gt_boxes[:, 6].unsqueeze(0))  # (num_queries, num_gt)
        
        return (
            self.class_weight * class_cost +
            self.center_weight * center_cost +
            self.size_weight * size_cost +
            self.angle_weight * angle_cost
        )

    def _compute_loss(self, pred_logits, pred_center, pred_size, pred_angle, gt_labels, gt_boxes, row_ind, col_ind):
        # Classification loss
        class_loss = F.cross_entropy(
            pred_logits[row_ind],  # (num_matched, num_classes)
            gt_labels[col_ind]     # (num_matched,)
        )
        
        # Center regression loss
        center_loss = F.l1_loss(
            pred_center[row_ind],   # (num_matched, 3)
            gt_boxes[col_ind, :3]   # (num_matched, 3)
        )
        
        # Size regression loss
        size_loss = F.l1_loss(
            pred_size[row_ind],     # (num_matched, 3)
            gt_boxes[col_ind, 3:6]  # (num_matched, 3)
        )
        
        # Angle regression loss
        angle_loss = F.l1_loss(
            pred_angle[row_ind],    # (num_matched, 1)
            gt_boxes[col_ind, 6].unsqueeze(1)  # (num_matched, 1)
        )
        
        return (
            self.class_weight * class_loss +
            self.center_weight * center_loss +
            self.size_weight * size_loss +
            self.angle_weight * angle_loss
        )

# Data Loading Utilities
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

def prepare_ground_truth(nusc, sample_token, class_map):
    sample = nusc.get('sample', sample_token)
    anns = [nusc.get('sample_annotation', token) for token in sample['anns']]
    
    boxes = []
    labels = []
    for ann in anns:
        if ann['category_name'] in class_map:
            # Extract (x, y, z, w, l, h, rot)
            box = ann['translation'] + ann['size'] + [ann['rotation'][0]]  # Only yaw rotation
            boxes.append(box)
            labels.append(class_map[ann['category_name']])
    
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),  # (num_gt, 7)
        'labels': torch.tensor(labels, dtype=torch.long)    # (num_gt,)
    }

# Training Loop
CLASS_MAP = {
    'human.pedestrian.adult': 0,
    'vehicle.car': 1,
    'vehicle.truck': 2,
    'vehicle.motorcycle': 3,
    'vehicle.bicycle': 4,
    'movable_object.barrier': 5,
    'vehicle.construction': 6
}

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 144  # Not divisible by 8
nhead = 8
model = Transformer3DETR(num_classes=len(CLASS_MAP), hidden_dim=hidden_dim, nhead=nhead).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = HungarianLoss(num_classes=len(CLASS_MAP))

# Dataset setup
nusc = NuScenes(version='v1.0-mini', dataroot='/home/farshad/Desktop/AI/OpenPCDet/data/nuscenes/v1.0-mini', verbose=False)

# Training
for epoch in range(10):
    for i, sample in enumerate(nusc.sample):
        sample_token = sample['token']
        
        # Load data
        lidar = load_lidar_data(nusc, sample_token).to(device)
        gt = prepare_ground_truth(nusc, sample_token, CLASS_MAP)
        gt_boxes = gt['boxes'].to(device)
        gt_labels = gt['labels'].to(device)
        
        # Forward pass
        outputs = model(lidar)

        # Compute loss for final predictions
        final_loss = criterion(outputs, [{'boxes': gt_boxes, 'labels': gt_labels}])

        # Compute loss for intermediate predictions
        aux_loss = 0
        for aux_logits, aux_boxes in zip(outputs['aux_outputs']['pred_logits'], outputs['aux_outputs']['pred_boxes']):
            aux_loss += criterion({'pred_logits': aux_logits, 'pred_boxes': aux_boxes}, [{'boxes': gt_boxes, 'labels': gt_labels}])

        # Total loss
        total_loss = final_loss + 0.5 * aux_loss  # Weight