# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat,shapes_to_tensor
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks, Boxes

POINTREND_HEAD_REGISTRY = Registry("POINT_HEAD")
POINTREND_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def generate_regular_grid_point_coords(R, side_size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device)
    r = F.affine_grid(aff, torch.Size((1, 1, side_size, side_size)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


def point_sample_fine_grained_features(features_list, feature_scales, boxes, point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.
        boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
            together.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    cat_boxes = Boxes.cat(boxes)
    num_boxes = [b.tensor.size(0) for b in boxes]

    point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor, point_coords)
    split_point_coords_wrt_image = torch.split(point_coords_wrt_image, num_boxes)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = shapes_to_tensor([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale.to(feature_map.device)
            point_features_per_image.append(
                point_sample(
                    feature_map[idx_img].unsqueeze(0),
                    point_coords_scaled.unsqueeze(0),
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(1, 0)
            )
        point_features.append(cat(point_features_per_image, dim=1))

    return cat(point_features, dim=0), point_coords_wrt_image


def get_point_coords_wrt_image(boxes_coords, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.

    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
        point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
        point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]
    return point_coords_wrt_image


def sample_point_labels(instances, point_coords):
    """
    Sample point labels from ground truth mask given point_coords.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. So, i_th elememt of the list contains R_i objects and R_1 + ... + R_N is
            equal to R. The ground-truth gt_masks in each instance will be used to compute labels.
        points_coords (Tensor): A tensor of shape (R, P, 2), where R is the total number of
            instances and P is the number of points for each instance. The coordinates are in
            the absolute image pixel coordinate space, i.e. [0, H] x [0, W].

    Returns:
        Tensor: A tensor of shape (R, P) that contains the labels of P sampled points.
    """
    with torch.no_grad():
        gt_mask_logits = []
        point_coords_splits = torch.split(
            point_coords, [len(instances_per_image) for instances_per_image in instances]
        )
        for i, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            #assert isinstance(
            #    instances_per_image.gt_masks, BitMasks
            #), "Point head works with GT in 'bitmask' format. Set INPUT.MASK_FORMAT to 'bitmask'."
            #print(instances_per_image.image_width, instances_per_image.image_height)
            h,w = instances_per_image.image_size
            if not isinstance(instances_per_image.gt_masks, BitMasks):
                gt_bit_masks = BitMasks.from_polygon_masks(instances_per_image.gt_masks, h, w).tensor.to(point_coords_splits[i].device)
            else:
                gt_bit_masks = instances_per_image.gt_masks.tensor
            
            scale = torch.tensor([w, h], dtype=torch.float, device=gt_bit_masks.device)
            points_coord_grid_sample_format = point_coords_splits[i] / scale
            gt_mask_logits.append(
                point_sample(
                    gt_bit_masks.to(torch.float32).unsqueeze(1),
                    points_coord_grid_sample_format,
                    align_corners=False,
                ).squeeze(1)
            )

    point_labels = cat(gt_mask_logits)
    return point_labels



def roi_mask_point_loss(mask_logits, instances, point_labels):
    """
    Compute the point-based loss for instance segmentation mask predictions
    given point-wise mask prediction and its corresponding point-wise labels.
    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the `mask_logits`. So, i_th
            elememt of the list contains R_i objects and R_1 + ... + R_N is equal to R.
            The ground-truth labels (class, box, mask, ...) associated with each instance are stored
            in fields.
        point_labels (Tensor): A tensor of shape (R, P), where R is the total number of
            predicted masks and P is the number of points for each mask.
            Labels with value of -1 will be ignored.
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)

        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

    gt_mask_logits = point_labels
    point_ignores = point_labels == -1
    if gt_mask_logits.shape[0] == 0:
        return mask_logits.sum() * 0

    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.to(dtype=torch.uint8)
    mask_accurate = mask_accurate[~point_ignores]
    mask_accuracy = mask_accurate.nonzero().size(0) / max(mask_accurate.numel(), 1.0)
    get_event_storage().put_scalar("point/accuracy", mask_accuracy)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), weight=~point_ignores, reduction="mean"
    )
    return point_loss

def calculate_uncertainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -(torch.abs(gt_class_logits))


@POINTREND_HEAD_REGISTRY.register()
class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        num_classes                 = cfg.MODEL.POINT_HEAD.NUM_CLASSES
        fc_dim                      = cfg.MODEL.POINT_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC
        cls_agnostic_mask           = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        input_channels              = input_shape.channels
        # fmt: on

        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)


@POINTREND_HEAD_REGISTRY.register()
class ImplicitPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(ImplicitPointHead, self).__init__()
        # fmt: off
        self.num_layers                         = cfg.MODEL.POINT_HEAD.NUM_FC + 1
        self.channels                           = cfg.MODEL.POINT_HEAD.FC_DIM
        self.image_feature_enabled              = cfg.MODEL.IMPLICIT_POINTREND.IMAGE_FEATURE_ENABLED
        self.positional_encoding_enabled        = cfg.MODEL.IMPLICIT_POINTREND.POS_ENC_ENABLED
        self.num_classes                        = (
            cfg.MODEL.POINT_HEAD.NUM_CLASSES if not cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK else 1
        )
        self.in_channels                        = input_shape.channels
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += 256
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, 128)))

        assert self.in_channels > 0

        num_weight_params, num_bias_params = [], []
        assert self.num_layers >= 2
        for l in range(self.num_layers):
            if l == 0:
                # input layer
                num_weight_params.append(self.in_channels * self.channels)
                num_bias_params.append(self.channels)
            elif l == self.num_layers - 1:
                # output layer
                num_weight_params.append(self.channels * self.num_classes)
                num_bias_params.append(self.num_classes)
            else:
                # intermediate layer
                num_weight_params.append(self.channels * self.channels)
                num_bias_params.append(self.channels)

        self.num_weight_params = num_weight_params
        self.num_bias_params = num_bias_params
        self.num_params = sum(num_weight_params) + sum(num_bias_params)

    def forward(self, fine_grained_features, point_coords, parameters):
        # features: [R, channels, K]
        # point_coords: [R, K, 2]
        num_instances = fine_grained_features.size(0)
        num_points = fine_grained_features.size(2)

        if num_instances == 0:
            return torch.zeros((0, 1, num_points), device=fine_grained_features.device)

        if self.positional_encoding_enabled:
            # locations: [R*K, 2]
            locations = 2 * point_coords.reshape(num_instances * num_points, 2) - 1
            locations = locations @ self.positional_encoding_gaussian_matrix.to(locations.device)
            locations = 2 * np.pi * locations
            locations = torch.cat([torch.sin(locations), torch.cos(locations)], dim=1)
            # locations: [R, C, K]
            locations = locations.reshape(num_instances, num_points, 256).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        weights, biases = self._parse_params(
            parameters,
            self.in_channels,
            self.channels,
            self.num_classes,
            self.num_weight_params,
            self.num_bias_params,
        )

        point_logits = self._dynamic_mlp(mask_feat, weights, biases, num_instances)
        point_logits = point_logits.reshape(-1, self.num_classes, num_points)

        return point_logits

    @staticmethod
    def _dynamic_mlp(features, weights, biases, num_instances):
        assert features.dim() == 3, features.dim()
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.einsum("nck,ndc->ndk", x, w) + b
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @staticmethod
    def _parse_params(
        pred_params,
        in_channels,
        channels,
        num_classes,
        num_weight_params,
        num_bias_params,
    ):
        assert pred_params.dim() == 2
        assert len(num_weight_params) == len(num_bias_params)
        assert pred_params.size(1) == sum(num_weight_params) + sum(num_bias_params)

        num_instances = pred_params.size(0)
        num_layers = len(num_weight_params)

        params_splits = list(
            torch.split_with_sizes(pred_params, num_weight_params + num_bias_params, dim=1)
        )

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l == 0:
                # input layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, in_channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            elif l < num_layers - 1:
                # intermediate layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            else:
                # output layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, num_classes, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, num_classes, 1)

        return weight_splits, bias_splits


def build_pointrend_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINTREND_HEAD_REGISTRY.get(head_name)(cfg, input_channels)
