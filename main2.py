import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import math
from typing import Optional
from torch import Tensor
# Add this import at the top of three_detr2.py
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes

# Add the build_preencoder function
def build_preenocder(preenc_npoints=1024, enc_dim=256):
    mlp_dims = [0, 64, 128, enc_dim]  # Assumes only XYZ input (no color)
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder
# Key Additions for Better Performance
class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(self, d_pos=256):
        super().__init__()
        self.d_pos = d_pos
        self.freq_bands = torch.nn.Parameter(
            (torch.arange(0, d_pos//6) / (d_pos//6 - 1)) * 2 * math.pi,
            requires_grad=False
        )
        
    def forward(self, xyz, input_range):
        # Normalize coordinates
        xyz = (xyz - input_range[0]) / (input_range[1] - input_range[0])
        
        # Sine/cosine embedding for each coordinate
        embeddings = []
        for coord in [xyz[..., 0], xyz[..., 1], xyz[..., 2]]:
            sins = torch.sin(coord.unsqueeze(-1) * self.freq_bands)
            coses = torch.cos(coord.unsqueeze(-1) * self.freq_bands)
            embeddings.append(torch.cat([sins, coses], dim=-1))
            
        return torch.cat(embeddings, dim=-1)
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
# ----------------------
# 2. Transformer Architecture (MODIFIED)
# ----------------------
class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, query_pos=None, mem_pos=None):  # CHANGED
        # Self attention with query pos
        q = k = self.with_pos_embed(tgt, query_pos)  # ADDED POS
        tgt2 = self.self_attn(
            q, k, value=tgt,  # USE POS-ENHANCED Q/K
            attn_mask=None,
            key_padding_mask=None
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention with POS-AWARE keys
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),  # QUERY POS
            key=self.with_pos_embed(memory, mem_pos),    # MEMORY POS
            value=memory,                                # RAW FEATURES
            key_padding_mask=None
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN (unchanged)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class CustomTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, query_pos=None, mem_pos=None):  # CHANGED
        output = tgt
        for mod in self.layers:
            # Pass both query and memory positions
            output = mod(output, memory, query_pos=query_pos, mem_pos=mem_pos)  # CHANGED
        return output



from functools import partial


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=128,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.use_norm_fn_on_input:
            src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [Tensor] = False):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=value, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st
WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        xyz_inds = None

        return xyz, output, xyz_inds

class Enhanced3DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100):
        super().__init__()

        
        self.preencoder = build_preenocder()
        # 2. Encoder-Decoder with positional embeddings
        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
        #     num_layers=4
        # )

        # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,  # Adjust as needed
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            norm_name="ln",
            use_ffn=True,
            ffn_use_bias=True
        )

        self.encoder = TransformerEncoder(encoder_layer, num_layers=4)

        # 3. Query generation using FPS + positional embeddings
        self.num_queries = num_queries
        self.query_pos = PositionEmbeddingCoordsSine(d_pos=hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # # 🔥 Add encoder positional embedding
        # self.enc_pos = PositionEmbeddingCoordsSine(d_pos=hidden_dim)
        
        # 🔥 Modify decoder layer for proper cross-attention
        self.decoder = CustomTransformerDecoder(
            CustomDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=4
        )
        # Add encoder positional embedding
        self.enc_pos_embed = PositionEmbeddingCoordsSine(d_pos=hidden_dim)
        

        # 5. Prediction heads with angle binning
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.center_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        self.size_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        self.angle_cls = nn.Linear(hidden_dim, 12)  # 12 angle bins
        self.angle_reg = nn.Linear(hidden_dim, 12)  # residuals per bin

    def forward(self, x):
        # 1. Get features and positions
        
        points = x
        xyz = points[..., 0:3].contiguous()
        features = points[..., 3:].transpose(1, 2).contiguous() if points.size(-1) > 3 else None
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.preencoder(xyz)
        # B = x.shape[0]
        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # # 2. Encoder processing
        # encoded = self.encoder(pre_enc_features)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )


        # enc_xyz, enc_features, enc_inds = self.encoder(pre_enc_features)
        
        # 3. Query generation using FPS
        query_xyz = self.farthest_point_sample(points, self.num_queries)
        pos_embed = self.query_pos(query_xyz, input_range=(x.min(1)[0], x.max(1)[0]))
        query_embed = self.query_proj(pos_embed)
        
        # Add encoder positional embeddings
        input_range = (x.min(dim=1)[0], x.max(dim=1)[0])
        enc_pos = self.enc_pos_embed(points, input_range)
        encoded = encoded + enc_pos  # Position-aware encoder features
        
        # Generate queries
        query_xyz = self.farthest_point_sample(points, self.num_queries)
        query_pos = self.query_pos(query_xyz, input_range)
        query_embed = self.query_proj(query_pos)
        
        # Decoder call with BOTH positions
        decoded = self.decoder(
            tgt=query_embed,
            memory=encoded,
            query_pos=query_pos,  # PASS QUERY POS
            mem_pos=enc_pos       # PASS MEMORY POS
        )
        
        
        # 5. Box predictions with angle binning
        outputs = {
            'class': self.class_head(decoded),
            'center': self.center_head(decoded),
            'size': self.size_head(decoded),
            'angle_cls': self.angle_cls(decoded),
            'angle_res': self.angle_reg(decoded)
        }
        return outputs

    def farthest_point_sample(self, xyz, n_samples):
        # Simplified FPS implementation
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, n_samples, device=xyz.device, dtype=torch.long)
        distances = torch.ones(B, N, device=xyz.device) * 1e10
        
        # Random initial point
        farthest = torch.randint(0, N, (B,), device=xyz.device)
        
        for i in range(n_samples):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = torch.max(distances, -1)[1]
            
        return torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))




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


def prepare_ground_truth(nusc, sample_token, class_map, num_angle_bins=12):
    sample = nusc.get('sample', sample_token)
    anns = [nusc.get('sample_annotation', token) for token in sample['anns']]
    
    # Initialize lists for batch=1
    labels = []
    centers = []
    sizes = []
    angle_bins = []
    angle_res = []
    
    for ann in anns:
        if ann['category_name'] not in class_map:
            continue
            
        # Box parameters
        center = torch.tensor(ann['translation'], dtype=torch.float32)
        size = torch.tensor(ann['size'], dtype=torch.float32)
        yaw = torch.tensor(ann['rotation'][-1])  # Assuming yaw rotation
        
        # Angle binning
        angle_bin_size = 2 * np.pi / num_angle_bins
        angle_norm = yaw % (2 * np.pi)
        bin_idx = int(angle_norm / angle_bin_size)
        residual = angle_norm - (bin_idx * angle_bin_size + angle_bin_size/2)
        
        labels.append(class_map[ann['category_name']])
        centers.append(center)
        sizes.append(size)
        angle_bins.append(bin_idx)
        angle_res.append(residual)
    
    # Return as lists to handle variable numbers of objects
    return {
        'labels': [torch.tensor(labels, dtype=torch.long)],
        'centers': [torch.stack(centers) if centers else torch.empty(0, 3)],
        'sizes': [torch.stack(sizes) if sizes else torch.empty(0, 3)],
        'angle_bins': [torch.tensor(angle_bins, dtype=torch.long)],
        'angle_res': [torch.tensor(angle_res, dtype=torch.float32)]
    }




def detection_loss(outputs, targets):
    """
    outputs: Dictionary with:
        - 'class': (B, num_queries, num_classes+1)
        - 'center': (B, num_queries, 3)
        - 'size': (B, num_queries, 3)
        - 'angle_cls': (B, num_queries, num_bins)
        - 'angle_res': (B, num_queries, num_bins)
        
    targets: Dictionary with:
        - 'labels': List[Tensor] (length B) of (num_gt_boxes,)
        - 'centers': List[Tensor] (length B) of (num_gt_boxes, 3)
        - 'sizes': List[Tensor] (length B) of (num_gt_boxes, 3)
        - 'angle_bins': List[Tensor] (length B) of (num_gt_boxes,)
        - 'angle_res': List[Tensor] (length B) of (num_gt_boxes,)
    """
    batch_size = outputs['class'].shape[0]
    total_loss = 0
    
    for batch_idx in range(batch_size):
        # Get predictions for this sample
        pred_logits = outputs['class'][batch_idx]  # (num_queries, num_classes+1)
        pred_center = outputs['center'][batch_idx]  # (num_queries, 3)
        pred_size = outputs['size'][batch_idx]      # (num_queries, 3)
        pred_angle_cls = outputs['angle_cls'][batch_idx]  # (num_queries, num_bins)
        pred_angle_res = outputs['angle_res'][batch_idx]  # (num_queries, num_bins)
        
        # Get ground truth for this sample
        gt_labels = targets['labels'][batch_idx]  # (num_gt,)
        gt_center = targets['centers'][batch_idx]  # (num_gt, 3)
        gt_size = targets['sizes'][batch_idx]      # (num_gt, 3)
        gt_angle_bin = targets['angle_bins'][batch_idx]  # (num_gt,)
        gt_angle_res = targets['angle_res'][batch_idx]  # (num_gt,)
        
        # Create cost matrix
        cost_class = -pred_logits[:, gt_labels].softmax(dim=-1)  # (num_queries, num_gt)
        cost_center = torch.cdist(pred_center, gt_center, p=1)  # (num_queries, num_gt)
        cost_size = torch.cdist(pred_size, gt_size, p=1)        # (num_queries, num_gt)
        
        # Combine costs with weights
        C = 1.0 * cost_class + 1.0 * cost_center + 0.5 * cost_size
        
        # Hungarian matching
        indices = linear_sum_assignment(C.detach().cpu().numpy())
        row_idx = torch.as_tensor(indices[0], device=pred_logits.device)
        col_idx = torch.as_tensor(indices[1], device=pred_logits.device)
        
        # Classification target (background class is last)
        background_class = outputs['class'].shape[-1] - 1
        target_labels = torch.full((pred_logits.size(0),), 
                                 background_class,
                                 dtype=torch.long,
                                 device=pred_logits.device)
        target_labels[row_idx] = gt_labels[col_idx]
        
        # Classification loss
        cls_loss = F.cross_entropy(pred_logits, target_labels)
        
        # Regression losses (only for matched pairs)
        center_loss = F.l1_loss(pred_center[row_idx], gt_center[col_idx])
        size_loss = F.l1_loss(pred_size[row_idx], gt_size[col_idx])
        
        # Angle losses
        angle_cls_loss = F.cross_entropy(pred_angle_cls[row_idx], gt_angle_bin[col_idx])
        angle_res_loss = F.l1_loss(
            pred_angle_res[row_idx, gt_angle_bin[col_idx]], 
            gt_angle_res[col_idx]
        )
        
        total_loss += (
            cls_loss + center_loss + size_loss + 
            angle_cls_loss + angle_res_loss
        )
        
    return total_loss / batch_size
# ----------------------
# 5. Training Loop
# ----------------------
# Configuration
# ----------------------
# 5. Training Loop (FIXED)
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
hidden_dim = 256
model = Enhanced3DETR(num_classes=len(CLASS_MAP), hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 🔥 Use the correct loss function
criterion = detection_loss  # Now using the proper detection loss

# Dataset setup
nusc = NuScenes(version='v1.0-mini', dataroot='/home/farshad/Desktop/AI/OpenPCDet/data/nuscenes/v1.0-mini', verbose=False)


# Training
for epoch in range(500):
    for i, sample in enumerate(nusc.sample):
        sample_token = sample['token']
        
        # Load data
        lidar = load_lidar_data(nusc, sample_token).to(device)
        
        # Get ground truth (already contains tensors)
        gt = prepare_ground_truth(nusc, sample_token, CLASS_MAP)
        
        # Move tensors to device
        gt_tensors = {
            'labels': [t.to(device) for t in gt['labels']],
            'centers': [t.to(device) for t in gt['centers']],
            'sizes': [t.to(device) for t in gt['sizes']],
            'angle_bins': [t.to(device) for t in gt['angle_bins']],
            'angle_res': [t.to(device) for t in gt['angle_res']]
        }
        
        # Forward + loss
        outputs = model(lidar)
        loss = detection_loss(outputs, gt_tensors)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1} Sample {i+1}/{len(nusc.sample)} Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), f'3detr_epoch_{epoch}.pth')
torch.save(model.state_dict(), '3detr_final.pth')
