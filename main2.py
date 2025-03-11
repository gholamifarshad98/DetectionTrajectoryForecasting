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
from third_party.pointnet2.pointnet2_utils import furthest_point_sample




from engine import evaluate, train_one_epoch
from optimizer import build_optimizer
from criterion import build_criterion
from torch.utils.data import DataLoader, Dataset
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import multiprocessing


'''
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [256, 1, 512]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
The error you're encountering is related to an inplace operation modifying a tensor that is needed for gradient computation. Specifically, the error message indicates that a tensor involved in the backward pass has been modified inplace, which breaks the computation graph and prevents PyTorch from correctly computing gradients.
'''
import torch.autograd
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


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
# # Key Additions for Better Performance
# class PositionEmbeddingCoordsSine(nn.Module):
#     def __init__(self, d_pos=256):
#         super().__init__()
#         self.d_pos = d_pos
#         self.freq_bands = torch.nn.Parameter(
#             (torch.arange(0, d_pos//6) / (d_pos//6 - 1)) * 2 * math.pi,
#             requires_grad=False
#         )
        
#     def forward(self, xyz, input_range):
#         # Normalize coordinates
#         xyz = (xyz - input_range[0]) / (input_range[1] - input_range[0])
        
#         # Sine/cosine embedding for each coordinate
#         embeddings = []
#         for coord in [xyz[..., 0], xyz[..., 1], xyz[..., 2]]:
#             sins = torch.sin(coord.unsqueeze(-1) * self.freq_bands)
#             coses = torch.cos(coord.unsqueeze(-1) * self.freq_bands)
#             embeddings.append(torch.cat([sins, coses], dim=-1))
            
#         return torch.cat(embeddings, dim=-1)



# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st

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
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=False)  # Changed to inplace=False
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=False)  # Changed to inplace=False

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)  # Changed to inplace=False
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

def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, num_semcls, num_angle_bin):
        self.num_semcls = num_semcls  # Number of semantic classes (excluding background)
        self.num_angle_bin = num_angle_bin  # Number of angle bins

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # Special case for datasets with no rotation angle
            # We still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    # def box_parametrization_to_corners(
    #     self, box_center_unnorm, box_size_unnorm, box_angle
    # ):
    #     """
    #     Convert box parameters (center, size, angle) into box corners.
    #     This is a placeholder implementation. Replace it with your actual logic.
    #     """
    #     # Placeholder implementation: Replace with your actual logic
    #     # For example, you can use a predefined function or formula to compute corners.
    #     # Here, we return a dummy tensor of shape (8, 3) for demonstration purposes.
    #     return torch.tensor([
    #         [0, 0, 0],
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1],
    #         [1, 1, 0],
    #         [1, 0, 1],
    #         [0, 1, 1],
    #         [1, 1, 1],
    #     ], dtype=torch.float32)


    def box_parametrization_to_corners(self, box_center_unnorm, box_size_unnorm, box_angle):
        """
        Convert box parameters (center, size, angle) into box corners.
        
        Args:
            box_center_unnorm: Tensor of shape (B, K1, 3) containing the unnormalized box centers.
            box_size_unnorm: Tensor of shape (B, K1, 3) containing the unnormalized box sizes.
            box_angle: Tensor of shape (B, K1) containing the box angles.
        
        Returns:
            Tensor of shape (B, K1, 8, 3) containing the 3D corners of the boxes.
        """
        B, K1 = box_center_unnorm.shape[:2]
        
        # Define the 8 corners of a unit cube
        unit_corners = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ], dtype=torch.float32, device=box_center_unnorm.device)  # Shape: (8, 3)
        
        # Scale the unit cube by the box size
        box_size_unnorm = box_size_unnorm.unsqueeze(2)  # Shape: (B, K1, 1, 3)
        scaled_corners = unit_corners * box_size_unnorm  # Shape: (B, K1, 8, 3)
        
        # Rotate the corners by the box angle
        cos_angle = torch.cos(box_angle).unsqueeze(-1).unsqueeze(-1)  # Shape: (B, K1, 1, 1)
        sin_angle = torch.sin(box_angle).unsqueeze(-1).unsqueeze(-1)  # Shape: (B, K1, 1, 1)
        
        # Rotation matrix around the Z-axis
        rotation_matrix = torch.cat([
            torch.cat([cos_angle, -sin_angle, torch.zeros_like(cos_angle)], dim=-1),
            torch.cat([sin_angle, cos_angle, torch.zeros_like(cos_angle)], dim=-1),
            torch.cat([torch.zeros_like(cos_angle), torch.zeros_like(cos_angle), torch.ones_like(cos_angle)], dim=-1),
        ], dim=-2)  # Shape: (B, K1, 3, 3)
        
        # Apply rotation to the corners
        rotated_corners = torch.matmul(scaled_corners, rotation_matrix)  # Shape: (B, K1, 8, 3)
        
        # Translate the corners to the box center
        box_center_unnorm = box_center_unnorm.unsqueeze(2)  # Shape: (B, K1, 1, 3)
        box_corners = rotated_corners + box_center_unnorm  # Shape: (B, K1, 8, 3)
        
        return box_corners



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



class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)
        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        
        # Disable inplace for all Dropout layers
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)  # Disable inplace
        
        # Explicitly set inplace=False for ReLU
        self.activation = ACTIVATION_DICT[activation](inplace=False)  # Fix here
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional [bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm_fn_name="ln",
                return_intermediate=False,
                weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                transpose_swap: Optional [bool] = False,
                return_attn_weights: Optional [bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1) # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_attn_weights=return_attn_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns


class Enhanced3DETR(nn.Module):
    def __init__(self, num_classes, 
        hidden_dim=256,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,):
        super().__init__()

                # Hardcoded values for dataset_config
        self.num_semcls = num_classes  # Number of semantic classes (excluding background)
        self.num_angle_bin = 12  # Number of angle bins (adjust as needed)

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


        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[encoder_dim], # or [encoder_dim, encoder_dim] if hasattr(self.encoder, "masking_radius"): in line 104 of https://github.com/facebookresearch/3detr/blob/main/models/model_3detr.py
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )


        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )



        # decoder_layer = TransformerDecoderLayer(
        #     d_model=args.dec_dim,
        #     nhead=args.dec_nhead,
        #     dim_feedforward=args.dec_ffn_dim,
        #     dropout=args.dec_dropout,
        # )
        # self.decoder = TransformerDecoder(
        #     decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
        # )

        # Assuming args is a namespace or dictionary containing the required parameters
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,  # Use decoder_dim from the Enhanced3DETR class
            nhead=8,  # Number of attention heads (adjust as needed)
            dim_feedforward=512,  # Dimension of the feedforward network (adjust as needed)
            dropout=0.1,  # Dropout rate (adjust as needed)
            activation="relu",  # Activation function
            normalize_before=True,  # Whether to normalize before or after attention
            norm_fn_name="ln",  # Normalization function (e.g., LayerNorm)
        )

        self.decoder = TransformerDecoder(
            decoder_layer, 
            num_layers=4,  # Number of decoder layers (adjust as needed)
            return_intermediate=True  # Whether to return intermediate outputs
        )



        # Build MLP Heads
        self.build_mlp_heads(decoder_dim, mlp_dropout)

        ############################################
        # 3. Query generation using FPS + positional embeddings
        self.num_queries = num_queries
        # self.query_pos = PositionEmbeddingCoordsSine(d_pos=hidden_dim)
        # self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # # # 🔥 Add encoder positional embedding
        # # self.enc_pos = PositionEmbeddingCoordsSine(d_pos=hidden_dim)
        
        # # 🔥 Modify decoder layer for proper cross-attention
        # self.decoder = CustomTransformerDecoder(
        #     CustomDecoderLayer(
        #         d_model=hidden_dim,
        #         nhead=8,
        #         batch_first=True
        #     ),
        #     num_layers=4
        # )
        # # Add encoder positional embedding
        # self.enc_pos_embed = PositionEmbeddingCoordsSine(d_pos=hidden_dim)
        
        self.box_processor = BoxProcessor(
            num_semcls=self.num_semcls, num_angle_bin=self.num_angle_bin
        )



        # # 5. Prediction heads with angle binning
        # self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        # self.center_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        # self.size_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        # self.angle_cls = nn.Linear(hidden_dim, 12)  # 12 angle bins
        # self.angle_reg = nn.Linear(hidden_dim, 12)  # residuals per bin

    def forward(self, x):
        # 1. Get features and positions
        
        points = x['point_clouds']
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

        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        #### GET QUERY EMBEDDING FUNCTION
 
        # point_cloud_dims = [
        #     torch.tensor([[0, 0, 0]], device=device),  # Shape (1, 3) for a single point cloud
        #     torch.tensor([[1, 1, 1]], device=device),  # Shape (1, 3) for a single point cloud
        # ]

        point_cloud_dims_min, point_cloud_dims_max = self.compute_point_cloud_dims(pre_enc_xyz)
        point_cloud_dims = [point_cloud_dims_min, point_cloud_dims_max]



        query_inds = furthest_point_sample(enc_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(enc_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        ###### END OF THAT FUNCTION

        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)

        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]


        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )



        
        return box_predictions
        # return outputs

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

    def compute_point_cloud_dims(self,points):
        assert points.ndim == 3 and points.shape[0] == 1 and points.shape[2] == 3, \
            "Input tensor must have shape [1, N, 3]"

        point_cloud_dims_min = torch.amin(points, dim=1)  
        point_cloud_dims_max = torch.amax(points, dim=1) 
                
        return point_cloud_dims_min, point_cloud_dims_max

    def build_mlp_heads(self, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # Add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=self.num_semcls + 1)

        # Geometry of the box
        center_head = mlp_func(output_dim=3)  # 3D center (x, y, z)
        size_head = mlp_func(output_dim=3)  # 3D size (width, height, depth)
        angle_cls_head = mlp_func(output_dim=self.num_angle_bin)  # Angle classification
        angle_reg_head = mlp_func(output_dim=self.num_angle_bin)  # Angle regression

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)



    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                            min: batch x 3 tensor of min XYZ coords
                            max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # Ensure box_corners has the correct shape (B, K1, 8, 3)
            if len(box_corners.shape) == 3:  # Shape: (K1, 8, 3)
                box_corners = box_corners.unsqueeze(0)  # Add batch dimension: (1, K1, 8, 3)
            assert len(box_corners.shape) == 4 and box_corners.shape[2] == 8 and box_corners.shape[3] == 3, \
                f"box_corners has incorrect shape: {box_corners.shape}"

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,  # Shape: (B, K1, 8, 3)
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }


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


# def prepare_ground_truth(nusc, sample_token, class_map, num_angle_bins=12):
#     sample = nusc.get('sample', sample_token)
#     anns = [nusc.get('sample_annotation', token) for token in sample['anns']]
    
#     # Initialize lists for batch=1
#     labels = []
#     centers = []
#     sizes = []
#     angle_bins = []
#     angle_res = []
    
#     for ann in anns:
#         if ann['category_name'] not in class_map:
#             continue
            
#         # Box parameters
#         center = torch.tensor(ann['translation'], dtype=torch.float32)
#         size = torch.tensor(ann['size'], dtype=torch.float32)
#         yaw = torch.tensor(ann['rotation'][-1])  # Assuming yaw rotation
        
#         # Angle binning
#         angle_bin_size = 2 * np.pi / num_angle_bins
#         angle_norm = yaw % (2 * np.pi)
#         bin_idx = int(angle_norm / angle_bin_size)
#         residual = angle_norm - (bin_idx * angle_bin_size + angle_bin_size/2)
        
#         labels.append(class_map[ann['category_name']])
#         centers.append(center)
#         sizes.append(size)
#         angle_bins.append(bin_idx)
#         angle_res.append(residual)
    
#     # Return as lists to handle variable numbers of objects
#     return {
#         'labels': [torch.tensor(labels, dtype=torch.long)],
#         'centers': [torch.stack(centers) if centers else torch.empty(0, 3)],
#         'sizes': [torch.stack(sizes) if sizes else torch.empty(0, 3)],
#         'angle_bins': [torch.tensor(angle_bins, dtype=torch.long)],
#         'angle_res': [torch.tensor(angle_res, dtype=torch.float32)]
#     }
def prepare_ground_truth(nusc, sample_token, class_map, num_angle_bins=12):
    sample = nusc.get('sample', sample_token)
    anns = [nusc.get('sample_annotation', token) for token in sample['anns']]
    
    # Initialize lists for batch=1
    labels = []
    centers = []
    sizes = []
    angle_bins = []
    angle_res = []
    box_corners = []
    box_angles = []

    for ann in anns:
        if ann['category_name'] not in class_map:
            continue
            
        # Box parameters
        center = torch.tensor(ann['translation'], dtype=torch.float32)
        size = torch.tensor(ann['size'], dtype=torch.float32)
        yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]  # Extract yaw angle

        # Angle binning
        angle_bin_size = 2 * np.pi / num_angle_bins
        angle_norm = yaw % (2 * np.pi)
        bin_idx = int(angle_norm / angle_bin_size)
        residual = angle_norm - (bin_idx * angle_bin_size + angle_bin_size/2)
        
        # Compute box corners
        rotation = Quaternion(ann['rotation'])
        corners = np.array([
            [size[0] / 2, size[1] / 2, size[2] / 2],
            [size[0] / 2, size[1] / 2, -size[2] / 2],
            [size[0] / 2, -size[1] / 2, size[2] / 2],
            [size[0] / 2, -size[1] / 2, -size[2] / 2],
            [-size[0] / 2, size[1] / 2, size[2] / 2],
            [-size[0] / 2, size[1] / 2, -size[2] / 2],
            [-size[0] / 2, -size[1] / 2, size[2] / 2],
            [-size[0] / 2, -size[1] / 2, -size[2] / 2],
        ])
        
        # Rotate each corner individually
        rotated_corners = np.array([rotation.rotate(corner) for corner in corners])
        rotated_corners += center.numpy()  # Translate to the box center
        rotated_corners = torch.tensor(rotated_corners, dtype=torch.float32)

        labels.append(class_map[ann['category_name']])
        centers.append(center)
        sizes.append(size)
        angle_bins.append(bin_idx)
        angle_res.append(residual)
        box_corners.append(rotated_corners)
        box_angles.append(yaw)
    
    # Convert lists to tensors
    labels = torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long)
    centers = torch.stack(centers) if centers else torch.empty(0, 3)
    sizes = torch.stack(sizes) if sizes else torch.empty(0, 3)
    angle_bins = torch.tensor(angle_bins, dtype=torch.long) if angle_bins else torch.empty(0, dtype=torch.long)
    angle_res = torch.tensor(angle_res, dtype=torch.float32) if angle_res else torch.empty(0, dtype=torch.float32)
    box_corners = torch.stack(box_corners) if box_corners else torch.empty(0, 8, 3)
    box_angles = torch.tensor(box_angles, dtype=torch.float32) if box_angles else torch.empty(0, dtype=torch.float32)

    # Compute point cloud dimensions
    lidar_data = load_lidar_data(nusc, sample_token)
    point_cloud_dims_min = lidar_data.min(dim=1)[0]  # Shape: (1, 3)
    point_cloud_dims_max = lidar_data.max(dim=1)[0]  # Shape: (1, 3)

    # Normalize centers and sizes
    centers_normalized = (centers - point_cloud_dims_min) / (point_cloud_dims_max - point_cloud_dims_min)
    sizes_normalized = sizes / (point_cloud_dims_max - point_cloud_dims_min)

    # Add "gt_box_present" key
    gt_box_present = torch.ones(1, dtype=torch.float32)  # Assuming one box per sample

    # Add "gt_box_sem_cls_label" key
    gt_box_sem_cls_label = labels  # Use the labels tensor directly
    
    return {
        'labels': labels,
        'centers': centers,
        'sizes': sizes,
        'angle_bins': angle_bins,
        'angle_res': angle_res,
        'point_cloud_dims_min': point_cloud_dims_min,
        'point_cloud_dims_max': point_cloud_dims_max,
        'gt_box_present': gt_box_present,
        'gt_box_corners': box_corners,
        'gt_box_centers_normalized': centers_normalized,
        'gt_box_sizes_normalized': sizes_normalized,
        'gt_angle_class_label': angle_bins,
        'gt_angle_residual_label': angle_res,
        'gt_box_angles': box_angles,
        'gt_box_sem_cls_label': gt_box_sem_cls_label,  # Add this key
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



#######################################

import argparse


def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##soheil!
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")


    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    # parser.add_argument(
    #     "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"]
    # )

    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser






import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import logging

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Initialize model, optimizer, and criterion
parser = make_args_parser()
args = parser.parse_args()
hidden_dim = 256
model = Enhanced3DETR(num_classes=len(CLASS_MAP), hidden_dim=hidden_dim).to(device)
num_classes = len(CLASS_MAP)
criterion = build_criterion(args, num_semcls=num_classes, num_angle_bin=12).to(device)
model_no_ddp = model
optimizer = build_optimizer(args, model_no_ddp)

# Dataset setup
nusc = NuScenes(version='v1.0-mini', dataroot='/home/farshad/Desktop/AI/OpenPCDet/data/nuscenes/v1.0-mini', verbose=False)

class DatasetConfig:
    def __init__(self):
        self.class2type = {
            0: 'human.pedestrian.adult',
            1: 'vehicle.car',
            2: 'vehicle.truck',
            3: 'vehicle.motorcycle',
            4: 'vehicle.bicycle',
            5: 'movable_object.barrier',
            6: 'vehicle.construction'
        }
        self.num_classes = len(self.class2type)

dataset_config = DatasetConfig()

# Dataset class
class NuScenesDataset(Dataset):
    def __init__(self, nusc, CLASS_MAP):
        self.nusc = nusc
        self.CLASS_MAP = CLASS_MAP
        self.samples = nusc.sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]['token']
        lidar = load_lidar_data(self.nusc, sample_token)
        gt = prepare_ground_truth(self.nusc, sample_token, self.CLASS_MAP)
        
        # Move all tensors to the correct device
        lidar = lidar.to(device)
        gt = {
            'labels': gt['labels'].to(device),
            'centers': gt['centers'].to(device),
            'sizes': gt['sizes'].to(device),
            'angle_bins': gt['angle_bins'].to(device),
            'angle_res': gt['angle_res'].to(device),
            'point_cloud_dims_min': gt['point_cloud_dims_min'].to(device),
            'point_cloud_dims_max': gt['point_cloud_dims_max'].to(device),
            'gt_box_present': gt['gt_box_present'].to(device),
            'gt_box_corners': gt['gt_box_corners'].to(device),
            'gt_box_centers_normalized': gt['gt_box_centers_normalized'].to(device),
            'gt_box_sizes_normalized': gt['gt_box_sizes_normalized'].to(device),
            'gt_angle_class_label': gt['gt_angle_class_label'].to(device),
            'gt_angle_residual_label': gt['gt_angle_residual_label'].to(device),
            'gt_box_angles': gt['gt_box_angles'].to(device),
            'gt_box_sem_cls_label': gt['gt_box_sem_cls_label'].to(device),
        }
        
        return {
            'point_clouds': lidar.squeeze(0),  # Remove batch dimension
            **gt  # Unpack all ground truth tensors
        }


import wandb
class Logger:
    def __init__(self):
        wandb.init(project="my-project")  # Initialize WandB

    def log_scalars(self, scalars, step, prefix=""):
        wandb.log({f"{prefix}{key}": value for key, value in scalars.items()}, step=step)

# Main execution block
if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')

    # Optional: Call freeze_support() if running on Windows or freezing to an executable
    multiprocessing.freeze_support()

    # Create dataset and dataloader
    nusc_dataset = NuScenesDataset(nusc, CLASS_MAP)
    train_dataloader = DataLoader(nusc_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # # Set up logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger("train")

    logger = Logger()  # Use the custom Logger class

    # Training loop
    for epoch in range(args.start_epoch, args.max_epoch):
        print(f"Epoch {epoch + 1}/{args.max_epoch}")
        print(f"Model device: {next(model.parameters()).device}")

        aps = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            train_dataloader,
            logger,
        )
        print(f"Epoch {epoch + 1} completed. APs: {aps}")
