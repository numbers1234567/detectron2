import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import PolygonMasks
from detectron2.layers import cat
from detectron2.utils.registry import Registry
import pycocotools.mask as mask_util

import copy

ROI_MASKIOU_HEAD_REGISTRY = Registry("ROI_MASKIOU_HEAD")
ROI_MASKIOU_HEAD_REGISTRY.__doc__ = """
Registry for maskiou heads, which predicts predicted mask iou.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def _crop(self : PolygonMasks, polygons: np.ndarray, box: np.ndarray) -> List[np.ndarray]:
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
        p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)

    return polygons

def crop(self : PolygonMasks, boxes: torch.Tensor) -> PolygonMasks:
    boxes = boxes.to(torch.device("cpu")).numpy()
    results = [
        _crop(self, polygon, box) for polygon, box in zip(self.polygons, boxes)
    ]

    # print('origin: ', self.polygons[0][0])
    # print('cropped: ', results[0][0])
    return PolygonMasks(results)

def get_maskiou_head_inputs(pred_mask_logits, instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    mask_ratios = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        if not cls_agnostic_mask or True:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        cropped_mask = crop(instances_per_image.gt_masks, instances_per_image.proposal_boxes.tensor)
        cropped_mask = torch.tensor(
            [mask_util.area(mask_util.frPyObjects([p for p in obj], box[3]-box[1], box[2]-box[0])).sum().astype(float)
            for obj, box in zip(cropped_mask.polygons, instances_per_image.proposal_boxes.tensor)]
            )
            
        mask_ratios.append(
            (cropped_mask / instances_per_image.gt_masks.area())
            .to(device=pred_mask_logits.device).clamp(min=0., max=1.)
        )
            
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    gt_classes = cat(gt_classes, dim=0)

    if len(gt_masks) == 0:
        selected_index = torch.arange(pred_mask_logits.shape[0], device=pred_mask_logits.device)
        selected_mask = pred_mask_logits[selected_index, gt_classes]
        mask_num, mask_h, mask_w = selected_mask.shape
        selected_mask = selected_mask.reshape(mask_num, 1, mask_h, mask_w)
        return selected_mask, gt_classes, None

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        # gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
    
    mask_ratios = cat(mask_ratios, dim=0)
    value_eps = 1e-10 * torch.ones(gt_masks.shape[0], device=gt_classes.device)
    mask_ratios = torch.max(mask_ratios, value_eps)

    pred_masks = pred_mask_logits > 0
    
    mask_targets_full_area = gt_masks.sum(dim=[1,2]) / mask_ratios
    # mask_ovr = pred_masks * gt_masks
    mask_ovr_area = (pred_masks * gt_masks).sum(dim=[1,2]).float()
    mask_union_area = pred_masks.sum(dim=[1,2]) + mask_targets_full_area - mask_ovr_area
    value_1 = torch.ones(pred_masks.shape[0], device=gt_classes.device)
    value_0 = torch.zeros(pred_masks.shape[0], device=gt_classes.device)
    mask_union_area = torch.max(mask_union_area, value_1)
    mask_ovr_area = torch.max(mask_ovr_area, value_0)
    maskiou_targets = mask_ovr_area / mask_union_area
    # selected_index = torch.arange(pred_mask_logits.shape[0], device=gt_classes.device)
    # selected_mask = pred_mask_logits[selected_index, gt_classes]
    mask_num, mask_h, mask_w = pred_mask_logits.shape
    selected_mask = pred_mask_logits.reshape(mask_num, 1, mask_h, mask_w)
    selected_mask = selected_mask.sigmoid()
    
    return selected_mask, gt_classes, maskiou_targets.detach()


def mask_iou_loss(labels, pred_maskiou, gt_maskiou, loss_weight, loss_type):
    """
    Compute the maskiou loss.

    Args:
        labels (Tensor): Given mask labels
        pred_maskiou: Predicted maskiou
        gt_maskiou: Ground Truth IOU generated in mask head
    """
    def l2_loss(input, target):
        pos_inds = torch.nonzero(target > 0.0).squeeze(1)
        if pos_inds.shape[0] > 0:
            cond = torch.abs(input[pos_inds] - target[pos_inds])
            loss = 0.5 * cond**2 / pos_inds.shape[0]
        else:
            loss = input * 0.0
        return loss.sum()
    
    def smooth_l1_loss(input, target, beta=1. / 9):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss.mean()

    if labels.numel() == 0:
        return pred_maskiou.sum() * 0
    if loss_type=="Smooth L1": loss_func = smooth_l1_loss
    elif loss_type=="L2":      loss_func = l2_loss
    else: raise ValueError("loss_type must be either Smooth L1 or L2")
    loss_func = smooth_l1_loss if loss_type=="Smooth L1" else l2_loss
    
    index = torch.arange(pred_maskiou.shape[0]).to(device=pred_maskiou.device)
    maskiou_loss = loss_func(pred_maskiou[index, labels], gt_maskiou)
    maskiou_loss = loss_weight * maskiou_loss
    
    return maskiou_loss


def mask_iou_inference(pred_instances, pred_maskiou):
    labels = cat([i.pred_classes for i in pred_instances])
    num_masks = pred_maskiou.shape[0]
    index = torch.arange(num_masks, device=labels.device)
    num_boxes_per_image = [len(i) for i in pred_instances]
    maskious = pred_maskiou[index, labels].split(num_boxes_per_image, dim=0)
    for maskiou, box in zip(maskious, pred_instances):
        box.pred_ious = maskiou
        box.scores = box.scores * maskiou

@ROI_MASKIOU_HEAD_REGISTRY.register()
class BaseMaskIoUHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        input_channels = 257

        self.maskiou_fcn1 = nn.Conv2d(input_channels, 256, 3, 1, 1) 
        self.maskiou_fcn2 = nn.Conv2d(256, 256, 3, 1, 1) 
        self.maskiou_fcn3 = nn.Conv2d(256, 256, 3, 1, 1) 
        self.maskiou_fcn4 = nn.Conv2d(256, 256, 3, 2, 1) 
        self.maskiou_fc1 = nn.Linear(256*7*7, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)
        self.maskiou = nn.Linear(1024, num_classes)

        for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3, self.maskiou_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        x = self.maskiou(x)
        return x

def build_maskiou_head(cfg, input_shape):
    return BaseMaskIoUHead(cfg)