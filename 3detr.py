import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

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

# ----------------------
# 5. Training Loop
# ----------------------
# Configuration
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
model = Transformer3DETR(num_classes=len(CLASS_MAP)).to(device)
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
        
        # Forward
        outputs = model(lidar)
        
        # Loss
        loss = criterion(outputs, [{'boxes': gt_boxes, 'labels': gt_labels}])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1} Sample {i+1}/{len(nusc.sample)} Loss: {loss.item():.4f}')