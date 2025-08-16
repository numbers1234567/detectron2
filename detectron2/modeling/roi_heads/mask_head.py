# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List,Tuple
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.layers.wrappers import move_device_like
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
import numpy as np

__all__ = [
    "BaseMaskRCNNHead",
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "ROI_MASK_HEAD_REGISTRY",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and sigmoid(0.0) == 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        device = (
            class_pred.device
            if torch.jit.is_scripting()
            else ("cpu" if torch.jit.is_tracing() else class_pred.device)
        )
        indices = move_device_like(torch.arange(num_masks, device=device), class_pred)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0):
        """
        NOTE: this interface is experimental.

        Args:
            loss_weight (float): multiplier of the loss
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight

        self.x : torch.Tensor|None = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        self.x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(self.x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(self.x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x
    
@ROI_MASK_HEAD_REGISTRY.register()
class PointRendCoarseMask(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    _version = 2

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dim: int, fc_dims: List[int], output_shape: Tuple[int]
    ):
        """
        Args:
            conv_dim: the output dimension of the conv layers
            fc_dims: a list of N>0 integers representing the output dimensions of N FC layers
            output_shape: shape of the output mask prediction
        """
        super().__init__()

        # fmt: off
        input_channels    = input_shape.channels
        input_h           = input_shape.height
        input_w           = input_shape.width
        self.output_shape = output_shape
        # fmt: on

        self.conv_layers = []
        if input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                activation=F.relu,
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = Conv2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True, activation=F.relu
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * input_h * input_w
        input_dim //= 4

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = nn.Linear(input_dim, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = fc_dim

        output_dim = int(np.prod(self.output_shape))

        self.prediction = nn.Linear(fc_dims[-1], output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        output_shape = (
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION,
            cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION,
        )
        fc_dim = cfg.MODEL.ROI_MASK_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_MASK_HEAD.NUM_FC
        ret = dict(
            input_shape=input_shape,
            conv_dim=cfg.MODEL.ROI_MASK_HEAD.CONV_DIM,
            fc_dims=[fc_dim] * num_fc,
            output_shape=output_shape,
        )
        return ret

    def layers(self, x):
        N = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        output_shape = [N] + list(self.output_shape)
        return self.prediction(x).view(*output_shape)

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def boundary_loss_func(boundary_logits, gtmasks):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0

    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        boundary_targets = F.interpolate(
            boundary_targets, boundary_logits.shape[2:], mode='nearest')

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)
    return bce_loss + dice_loss


def boundary_preserving_mask_loss(
        pred_mask_logits,
        pred_boundary_logits,
        instances,
        vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    boundary_loss = boundary_loss_func(pred_boundary_logits, gt_masks)
    return mask_loss, boundary_loss
    
class BoundaryAttentionFusion(nn.Module):
    def __init__(
        self, 
        cfg, 
        in_shape=(14,14), 
        dropout=0.1,
    ):
        super(BoundaryAttentionFusion, self).__init__()
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM

        embed_dim = cfg.MODEL.BOUNDARY_MASK_HEAD.ATTENTION_DIM
        if embed_dim==None:
            embed_dim = conv_dim

        self.in_shape = in_shape

        self.other_to_self = nn.MultiheadAttention(
            embed_dim, 8, batch_first=True
        )
        self.pos_embed = nn.Parameter(torch.randn((in_shape[0]*in_shape[1], embed_dim)))

        self.q_drop = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(conv_dim, embed_dim)
        self.v_proj = nn.Linear(conv_dim, embed_dim)

        self.down_proj = nn.Identity() if conv_dim==embed_dim else nn.Linear(conv_dim, embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.out_norm = nn.LayerNorm(embed_dim)

        self.up_proj = nn.Identity() if conv_dim==embed_dim else nn.Linear(embed_dim, conv_dim)

    def forward(self, this_features, other_features):
        # Transform features to "token" format
        this_features = torch.transpose(this_features, 1, 3)
        this_features = torch.flatten(this_features, 1, 2)
        
        other_features = torch.transpose(other_features, 1, 3)
        other_features = torch.flatten(other_features, 1, 2)

        # Main computations following ViT
        this_features = self.down_proj(this_features)
        zp,attn_weights = self.other_to_self(
            self.q_norm(self.q_proj(self.q_drop(this_features + self.pos_embed))), 
            self.k_norm(self.k_proj(other_features)), 
            self.v_norm(self.v_proj(other_features)),
        )

        zp = zp + this_features

        z = self.ff(self.out_norm(zp)) + zp
        z = self.up_proj(z)

        # Return to original format using the inverse of flatten and transpose
        z = torch.unflatten(z, 1, self.in_shape)
        z = torch.transpose(z, 1, 3)
        return z

@ROI_MASK_HEAD_REGISTRY.register()
class BoundaryPreservingAttentionHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(BoundaryPreservingAttentionHead, self).__init__()

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_boundary_conv = cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1

        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.boundary_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_to_boundary = BoundaryAttentionFusion(
            cfg,
            (14, 14),
        )

        self.boundary_to_mask = BoundaryAttentionFusion(
            cfg,
            (14, 14),
        )

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self.boundary_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_fcns + self.boundary_fcns +\
                     [self.mask_deconv, self.boundary_deconv,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)

    def forward(self, mask_features, boundary_features, instances: List[Instances]):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        # downsample
        boundary_features = self.downsample(boundary_features)
        # mask to boundary fusion
        boundary_features = self.mask_to_boundary(boundary_features, mask_features)
        for layer in self.boundary_fcns:
            boundary_features = layer(boundary_features)
        # boundary to mask fusion
        mask_features = self.boundary_to_mask(mask_features, boundary_features)
        mask_features = self.mask_final_fusion(mask_features)
        # mask prediction
        mask_features = F.relu(self.mask_deconv(mask_features))
        mask_logits = self.mask_predictor(mask_features)
        # boundary prediction
        boundary_features = F.relu(self.boundary_deconv(boundary_features))
        boundary_logits = self.boundary_predictor(boundary_features)
        self.x = mask_logits
        if self.training:
            loss_mask, loss_boundary = boundary_preserving_mask_loss(
                mask_logits, boundary_logits, instances)
            return {"loss_mask": loss_mask,
                    "loss_boundary": loss_boundary}
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances
        
@ROI_MASK_HEAD_REGISTRY.register()
class BoundaryPreservingHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(BoundaryPreservingHead, self).__init__()

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_boundary_conv = cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1

        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.boundary_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_to_boundary = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.boundary_to_mask = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self.boundary_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_fcns + self.boundary_fcns +\
                     [self.mask_deconv, self.boundary_deconv, self.boundary_to_mask, self.mask_to_boundary,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)

    def forward(self, mask_features, boundary_features, instances: List[Instances]):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        # downsample
        boundary_features = self.downsample(boundary_features)
        # mask to boundary fusion
        boundary_features = boundary_features + self.mask_to_boundary(mask_features)
        for layer in self.boundary_fcns:
            boundary_features = layer(boundary_features)
        # boundary to mask fusion
        mask_features = self.boundary_to_mask(boundary_features) + mask_features
        mask_features = self.mask_final_fusion(mask_features)
        # mask prediction
        mask_features = F.relu(self.mask_deconv(mask_features))
        mask_logits = self.mask_predictor(mask_features)
        # boundary prediction
        boundary_features = F.relu(self.boundary_deconv(boundary_features))
        boundary_logits = self.boundary_predictor(boundary_features)
        self.x = mask_logits
        if self.training:
            loss_mask, loss_boundary = boundary_preserving_mask_loss(
                mask_logits, boundary_logits, instances)
            return {"loss_mask": loss_mask,
                    "loss_boundary": loss_boundary}
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances
        

def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
