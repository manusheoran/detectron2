
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads    
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork", "ProposalNetwork1", "ProposalNetwork_DA", "ProposalNetwork_DA_CA"]


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        #print("conv param", padding, stride)
        
        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
        
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)
        #self.conv_out = nn.Conv2d(self.in_channels, self.out_channels , self.kernel_size, stride=stride, padding=self.padding)
        
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
        
    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        x = x.reshape(-1, 5 * x.shape[1], x.shape[2], x.shape[3]) 
        conv_out = self.conv_out(x)
        
        batch, _, height, width = conv_out.size()
        #batch, _, height, width = x.size()
        #height= width=self.shape
        #print(conv_out.size())

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        #print("input to qkv", x.shape)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        #print("attn",attn_out.size())
        # combine_heads_2d
        # (batch, out_channels, height, width)
        #print("input to attn", attn_out.size())
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)
        
        #return conv_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        #print("qkv",qkv.size())
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

#augmented_conv_128 = AugmentedConv(in_channels=5*256, out_channels=256, kernel_size=3, dk=40, dv=4, Nh=2, relative=True,  stride=2, shape=64) 
augmented_conv_64 = AugmentedConv(in_channels=5*256, out_channels=256, kernel_size=3, dk=40, dv=4, Nh=2, relative=True,  stride=1, shape=64) 
augmented_conv_32 = AugmentedConv(in_channels=5*256, out_channels=256, kernel_size=3, dk=40, dv=4, Nh=2, relative=True, stride=1, shape=32) 
augmented_conv_16 = AugmentedConv(in_channels=5*256, out_channels=256, kernel_size=3, dk=40, dv=4, Nh=2, relative=True,  stride=1, shape=16) 
augmented_conv_8 = AugmentedConv(in_channels=5*256, out_channels=256, kernel_size=3, dk=40, dv=4, Nh=2, relative=True,  stride=1, shape=8) 
augmented_conv_4 = AugmentedConv(in_channels=5*256, out_channels=256, kernel_size=3, dk=40, dv=4, Nh=2, relative=True,  stride=1, shape=4) 

#attention with more dk dv = equal to convolution
# augmented_conv_128 = AugmentedConv(in_channels=3*256, out_channels=256, kernel_size=3, dk=64, dv=64, Nh=8, relative=True,  stride=2, shape=64) 
# augmented_conv_64 = AugmentedConv(in_channels=3*256, out_channels=256, kernel_size=3, dk=64, dv=64, Nh=8, relative=True,  stride=1, shape=64) 
# augmented_conv_32 = AugmentedConv(in_channels=3*256, out_channels=256, kernel_size=3, dk=64, dv=64, Nh=8, relative=True, stride=1, shape=32) 
# augmented_conv_16 = AugmentedConv(in_channels=3*256, out_channels=256, kernel_size=3, dk=64, dv=64, Nh=8, relative=True,  stride=1, shape=16) 
# augmented_conv_8 = AugmentedConv(in_channels=3*256, out_channels=256, kernel_size=3, dk=64, dv=64, Nh=8, relative=True,  stride=1, shape=8) 
#up_sample =  nn.ConvTranspose2d(256, 256, 3,stride=2, padding=1, output_padding=1)
@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

  
@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

    
@META_ARCH_REGISTRY.register()
class ProposalNetwork1(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.siz_div = build_backbone(cfg).size_divisibility     #fcos_byol
        self.out_shape_original = self.backbone.output_shape()      #fcos_byol
        #checkpoint = torch.load('/data/nihcc/BYOL/FCOS/norm_cropped_x101/byol_cropped_deeplesion000105.0.pt')     #fcos_byol
        #self.backbone.load_state_dict(checkpoint)     #fcos_byol
        #self.augmentedConv_128 = augmented_conv_128
        self.augmentedConv_64 = augmented_conv_64
        self.augmentedConv_32 = augmented_conv_32
        self.augmentedConv_16 = augmented_conv_16
        self.augmentedConv_8 = augmented_conv_8#AugmentedConv()
        self.augmentedConv_4 = augmented_conv_4
        #self.up_sample = up_sample

        self.proposal_generator = build_proposal_generator(cfg, self.out_shape_original)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN1).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD1).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        temp_images = ()
        for im in images:
            temp_images += im.split(3)

        images = ImageList.from_tensors(temp_images, self.siz_div)     #fcos_byol
        features = self.backbone(images.tensor)
#         for key in features.keys():
#         	print(key,features[key].shape)
        feature_fused = {}
        feature_fused['p3'] = self.augmentedConv_64(features['p3'])
        #print('feature_128.shape up sample',feature_fused['p2'].shape)
        feature_fused['p4'] = self.augmentedConv_32(features['p4'])
        feature_fused['p5'] = self.augmentedConv_16(features['p5'])
        feature_fused['p6'] = self.augmentedConv_8(features['p6'])
        feature_fused['p7'] = self.augmentedConv_4(features['p7'])
        
        my_image = images.tensor[3::5] #5 slice
        #my_image = images.tensor[4::9]
        #print(my_image.shape)
        my_image_sizes = [(my_image.shape[-2], my_image.shape[-1]) for im in my_image]
        #print(image_sizes)
        images = ImageList(my_image,my_image_sizes)


        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.proposal_generator(images, feature_fused, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

# class GradientReversalFunction(torch.autograd.Function):
#     """
#     Gradient Reversal Layer from:
#     Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
#     Forward pass is the identity function. In the backward pass,
#     the upstream gradients are multiplied by -_lambda (i.e. gradient is reversed)
#     """

#     @staticmethod
#     def forward(ctx, x, _lambda_):
#         ctx._lambda_ = _lambda_
#         return x.clone()

#     @staticmethod
#     def backward(ctx, grads):
#         _lambda_ = ctx._lambda_
#         _lambda_ = grads.new_tensor(_lambda_)
#         dx = -_lambda_ * grads
#         return dx, None


# class GradientReversal(torch.nn.Module):
#     def __init__(self, _lambda_=1):
#         super(GradientReversal, self).__init__()
#         self._lambda_ = _lambda_

#     def forward(self, x):
#         return GradientReversalFunction.apply(x, self._lambda_)
    
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None 

#CA_discriminator
class FCOSDiscriminator_CA(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, center_aware_weight=0.0, center_aware_type='ca_feature', grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_CA, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        #self.grad_reverse = GradReverse(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

        # hyperparameters
        assert center_aware_type == 'ca_loss' or center_aware_type == 'ca_feature'
        self.center_aware_weight = center_aware_weight
        self.center_aware_type = center_aware_type

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, _lambda_CA, score_map, domain):  
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        # Generate cneter-aware map
        box_cls_map = score_map["box_cls"].clone().sigmoid()
        centerness_map = score_map["centerness"].clone().sigmoid()

        n, c, h, w = box_cls_map.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map = maxpooling(box_cls_map)

        # Normalize the center-aware map
        atten_map = (self.center_aware_weight * box_cls_map * centerness_map).sigmoid()

        
        # Compute loss
        # Center-aware loss (w/ GRL)
        if self.center_aware_type == 'ca_loss':
            if self.grl_applied_domain == 'both':
                feature = GradReverse.apply( feature,_lambda_CA )
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = GradReverse.apply( feature,_lambda_CA )

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            # Computer loss
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn_no_reduce(x, target)
            loss = torch.mean(atten_map * loss)

        # Center-aware feature (w/ GRL)
        elif self.center_aware_type == 'ca_feature':
            if self.grl_applied_domain == 'both':
                feature = GradReverse.apply(atten_map * feature,_lambda_CA )
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = GradReverse.apply(atten_map * feature,_lambda_CA )

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x, target)

        return loss
    
    
#GA_Discriminator    
class FCOSDiscriminator(nn.Module):
    def __init__(self, num_convs=2, in_channels=256,  grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
#         for modules in [self.dis_tower, self.cls_logits]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#                     torch.nn.init.constant_(l.bias, 0)

        #self.grad_reverse = GradientReversal()
        self.loss_fn = nn.BCEWithLogitsLoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, _lambda, domain ):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        #print('inside discri\n', target)
        if self.grl_applied_domain == 'both':
            feature = GradReverse.apply(feature,_lambda )
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = GradReverse.apply(feature, _lambda)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        
        print(x.shape, np.sum(x))

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss

#dis_P7 = FCOSDiscriminator(num_convs=3, grl_applied_domain="both")#.to(device)    #not p7 
 
dis_P6 = FCOSDiscriminator(num_convs=4, grl_applied_domain="both")#.to(device)    #not p6 

dis_P5 = FCOSDiscriminator(num_convs=4, grl_applied_domain="both")#.to(device)

dis_P4 = FCOSDiscriminator(num_convs=4, grl_applied_domain="both")#.to(device)

dis_P3 = FCOSDiscriminator(num_convs=4, grl_applied_domain="both")#.to(device)

#dis_P7_CA = FCOSDiscriminator_CA(num_convs=4, center_aware_weight=20, grl_applied_domain='both')

dis_P6_CA = FCOSDiscriminator_CA(num_convs=4, center_aware_weight=20, grl_applied_domain='both')

dis_P5_CA = FCOSDiscriminator_CA(num_convs=4, center_aware_weight=20, grl_applied_domain='both')

dis_P4_CA = FCOSDiscriminator_CA(num_convs=4, center_aware_weight=20, grl_applied_domain='both')

dis_P3_CA = FCOSDiscriminator_CA(num_convs=4, center_aware_weight=20, grl_applied_domain='both')
    
@META_ARCH_REGISTRY.register()
class ProposalNetwork_DA(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        
        #DA FPN Layers :
        #self.dis_P7 = dis_P7    #not p7 
 
        self.dis_P6 = dis_P6     #not p6 

        self.dis_P5 = dis_P5

        self.dis_P4 = dis_P4

        self.dis_P3 = dis_P3
        
#         #CA 
#         #self.dis_P7 = dis_P7_CA    #not p7 
 
#         self.dis_P6_CA = dis_P6_CA     #not p6 

#         self.dis_P5_CA = dis_P5_CA

#         self.dis_P4_CA = dis_P4_CA

#         self.dis_P3_CA = dis_P3_CA
        
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    #DA Loss function
    def losses(self, images, f, gt_instances , _lambdas, domain_target):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        if domain_target:
            #loss_p7 = self.dis_P7(f['p7'], 1.0,_lambdas['p7'], domain='target')     #not p7 
            loss_p6 = self.dis_P6(f['p6'], 1.0, _lambdas['p6'], domain='target')     #not p6 
            loss_p5 = self.dis_P5(f['p5'], 1.0, _lambdas['p5'], domain='target') 
            loss_p4 = self.dis_P4(f['p4'], 1.0, _lambdas['p4'], domain='target') 
            loss_p3 = self.dis_P3(f['p3'], 1.0, _lambdas['p3'], domain='target') 
            
#             #CA losses
#             #loss_p7_CA = self.dis_P7_CA(f['p7'], 1.0,_lambdas_CA['p7'], domain='target')     #not p7 
#             loss_p6_CA = self.dis_P6_CA(f['p6'], 1.0, _lambdas_CA['p6'], domain='target')     #not p6 
#             loss_p5_CA = self.dis_P5_CA(f['p5'], 1.0, _lambdas_CA['p5'], domain='target') 
#             loss_p4_CA = self.dis_P4_CA(f['p4'], 1.0, _lambdas_CA['p4'], domain='target') 
#             loss_p3_CA = self.dis_P3_CA(f['p3'], 1.0, _lambdas_CA['p3'], domain='target')
            #proposal_losses = {"loss_p7": loss_p7,"loss_p6": loss_p6,"loss_p5": loss_p5,"loss_p4": loss_p4,"loss_p3": loss_p3}      #not p7     #not p6 
            proposal_losses = {"loss_p6": loss_p6,"loss_p5": loss_p5,"loss_p4": loss_p4,"loss_p3": loss_p3}
#             #CA combine proposals
#             proposal_losses = {"loss_p6": loss_p6,"loss_p5": loss_p5,"loss_p4": loss_p4,"loss_p3": loss_p3,"loss_p6_CA": loss_p6_CA,"loss_p5_CA": loss_p5_CA,
#                                "loss_p4_CA": loss_p4_CA,"loss_p3_CA": loss_p3_CA}
            proposals = {}
#             for name, layer in self.dis_P3.named_modules():
#                 if isinstance(layer, nn.Conv2d):
#                     if '.0' in name:
#                         print('for target',name, layer.weight.sum())
            return proposals, proposal_losses
            #return {"loss_r3": loss_res3, "loss_r4": loss_res4, "loss_r5": loss_res5}
        else:
            #loss_p7 = self.dis_P7(f['p7'], 0.0,_lambdas['p7'], domain='source')     #not p7 
            loss_p6 = self.dis_P6(f['p6'], 0.0, _lambdas['p6'], domain='source')     #not p6 
            loss_p5 = self.dis_P5(f['p5'], 0.0, _lambdas['p5'], domain='source') 
            loss_p4 = self.dis_P4(f['p4'], 0.0, _lambdas['p4'], domain='source') 
            loss_p3 = self.dis_P3(f['p3'], 0.0, _lambdas['p3'], domain='source') 
            
#             #CA losses
#             #loss_p7_CA = self.dis_P7_CA(f['p7'], 0.0,_lambdas_CA['p7'], domain='source')     #not p7 
#             loss_p6_CA = self.dis_P6_CA(f['p6'], 0.0, _lambdas_CA['p6'], domain='source')     #not p6 
#             loss_p5_CA = self.dis_P5_CA(f['p5'], 0.0, _lambdas_CA['p5'], domain='source') 
#             loss_p4_CA = self.dis_P4_CA(f['p4'], 0.0, _lambdas_CA['p4'], domain='source') 
#             loss_p3_CA = self.dis_P3_CA(f['p3'], 0.0, _lambdas_CA['p3'], domain='source') 
#             for name, layer in self.dis_P3.named_modules():
#                 if isinstance(layer, nn.Conv2d):
#                     if '.0' in name:
#                         print('for source',name, layer.weight.sum())
            
        #print('feature shape fp7 ', f['p7'].shape)
        proposals, proposal_losses, score_maps = self.proposal_generator(images, f, gt_instances)
        
        proposal_losses["loss_p3"] = loss_p3
        proposal_losses["loss_p4"] = loss_p4
        proposal_losses["loss_p5"] = loss_p5
        proposal_losses["loss_p6"] = loss_p6    #not p6 
        #proposal_losses["loss_p7"] = loss_p7     #not p7 
        
#         #CA proposals
#         proposal_losses["loss_p3_CA"] = loss_p3_CA
#         proposal_losses["loss_p4_CA"] = loss_p4_CA
#         proposal_losses["loss_p5_CA"] = loss_p5_CA
#         proposal_losses["loss_p6_CA"] = loss_p6_CA    #not p6 
#         #proposal_losses["loss_p7_CA"] = loss_p7_CA     #not p7 

        return proposals, proposal_losses  
    
    def forward(self, batched_inputs, _lambdas={}, domain_target = False ):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if len(_lambdas)==0:
            _lambdas['p3']=0.01 #0.5
            _lambdas['p4']=0.01#0.5
            _lambdas['p5']=0.01 #0.5
            _lambdas['p6']=0.01#0.1    #not p6 
            #_lambdas['p7']=0.01#0.1       #not p7
            
#             #CA lambdas
#             _lambdas_CA['p3']=0.02 #0.5
#             _lambdas_CA['p4']=0.02#0.5
#             _lambdas_CA['p5']=0.02 #0.5
#             _lambdas_CA['p6']=0.02#0.1    #not p6 
#             #_lambdas_CA['p7']=0.02#0.1       #not p7
            
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        #print('input shape', images.tensor.shape)
        features = self.backbone(images.tensor)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        #print("interanl lambdas", _lambdas)
        proposals, proposal_losses = self.losses(images, features, gt_instances , _lambdas, domain_target )
        
#         if domain_target:
#             print('target image', images.tensor.shape, images.tensor.sum())
#             print('for target p3 and p7 sum', features['p3'].sum(), features['p7'].sum() )
#             print('len of proposals for target', len(proposals))
            
#         if not domain_target:
#             print('source image', images.tensor.shape, images.tensor.sum())
#             print('for source p3 and p7 sum', features['p3'].sum() , features['p7'].sum() )
#             print('len of proposals for sourec', len(proposals))

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        
        #proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        #print(proposals)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses
        #print("\n running after training:")
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

   
@META_ARCH_REGISTRY.register()
class ProposalNetwork_DA_CA(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        
        #DA FPN Layers :
        #self.dis_P7 = dis_P7    #not p7 
 
        self.dis_P6 = dis_P6     #not p6 

        self.dis_P5 = dis_P5

        self.dis_P4 = dis_P4

        self.dis_P3 = dis_P3
        
#         #CA 
        #self.dis_P7 = dis_P7_CA    #not p7 
 
        self.dis_P6_CA = dis_P6_CA     #not p6 

        self.dis_P5_CA = dis_P5_CA

        self.dis_P4_CA = dis_P4_CA

        self.dis_P3_CA = dis_P3_CA
        
        self.proposal_generator = proposal_generator
        #self.proposal_generator_CA = proposal_generator_CA
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            #"proposal_generator_CA": build_proposal_generator_CA(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    #DA Loss function
    def losses(self, images, f, gt_instances , _lambdas, _lambdas_CA, domain_target):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        if domain_target:
            #loss_p7 = self.dis_P7(f['p7'], 1.0,_lambdas['p7'], domain='target')     #not p7 
            loss_p6 = self.dis_P6(f['p6'], 1.0, _lambdas['p6'], domain='target')     #not p6 
            loss_p5 = self.dis_P5(f['p5'], 1.0, _lambdas['p5'], domain='target') 
            loss_p4 = self.dis_P4(f['p4'], 1.0, _lambdas['p4'], domain='target') 
            loss_p3 = self.dis_P3(f['p3'], 1.0, _lambdas['p3'], domain='target') 
            
            score_maps= self.proposal_generator(images, f, gt_instances=None)
            
            map_layer_to_index = {"p3": 0, "p4": 1, "p5": 2, "p6": 3, "p7": 4}
            feature_layers = map_layer_to_index.keys()
            m = {
                layer: {
                    map_type:
                    score_maps[map_type][map_layer_to_index[layer]]
                    for map_type in score_maps
                }
                for layer in feature_layers
            }
#             #CA losses
            #loss_p7_CA = self.dis_P7_CA(f['p7'], 1.0,_lambdas_CA['p7'], domain='target')     #not p7 
            loss_p6_CA = self.dis_P6_CA(f['p6'], 1.0, _lambdas_CA['p6'],m['p6'], domain='target')     #not p6 
            loss_p5_CA = self.dis_P5_CA(f['p5'], 1.0, _lambdas_CA['p5'],m['p5'], domain='target') 
            loss_p4_CA = self.dis_P4_CA(f['p4'], 1.0, _lambdas_CA['p4'],m['p4'], domain='target') 
            loss_p3_CA = self.dis_P3_CA(f['p3'], 1.0, _lambdas_CA['p3'],m['p3'], domain='target')
            #proposal_losses = {"loss_p7": loss_p7,"loss_p6": loss_p6,"loss_p5": loss_p5,"loss_p4": loss_p4,"loss_p3": loss_p3}      #not p7     #not p6 
            #proposal_losses = {"loss_p6": loss_p6,"loss_p5": loss_p5,"loss_p4": loss_p4,"loss_p3": loss_p3}
#             #CA combine proposals
            proposal_losses = {"loss_p6": loss_p6,"loss_p5": loss_p5,"loss_p4": loss_p4,"loss_p3": loss_p3,"loss_p6_CA": loss_p6_CA,"loss_p5_CA": loss_p5_CA,
                               "loss_p4_CA": loss_p4_CA,"loss_p3_CA": loss_p3_CA}
            proposals = {}
#             for name, layer in self.dis_P3.named_modules():
#                 if isinstance(layer, nn.Conv2d):
#                     if '.0' in name:
#                         print('for target',name, layer.weight.sum())
            return proposals, proposal_losses
            #return {"loss_r3": loss_res3, "loss_r4": loss_res4, "loss_r5": loss_res5}
        else:
            #loss_p7 = self.dis_P7(f['p7'], 0.0,_lambdas['p7'], domain='source')     #not p7 
            loss_p6 = self.dis_P6(f['p6'], 0.0, _lambdas['p6'], domain='source')     #not p6 
            loss_p5 = self.dis_P5(f['p5'], 0.0, _lambdas['p5'], domain='source') 
            loss_p4 = self.dis_P4(f['p4'], 0.0, _lambdas['p4'], domain='source') 
            loss_p3 = self.dis_P3(f['p3'], 0.0, _lambdas['p3'], domain='source') 
            
            proposals, proposal_losses, score_maps= self.proposal_generator(images, f, gt_instances)
            map_layer_to_index = {"p3": 0, "p4": 1, "p5": 2, "p6": 3, "p7": 4}
            feature_layers = map_layer_to_index.keys()
            m = {
                layer: {
                    map_type:
                    score_maps[map_type][map_layer_to_index[layer]]
                    for map_type in score_maps
                }
                for layer in feature_layers
            }
            
#             for layer_name in m:
#               for map_name in m[layer_name]:
#                 print(layer_name, map_name, type(m[layer_name][map_name]),(m[layer_name][map_name].shape))
                    
#             print(type(f), type(_lambdas_CA), type(m) )
#             #CA losses
            #loss_p7_CA = self.dis_P7_CA(f['p7'], 1.0,_lambdas_CA['p7'], domain='target')     #not p7 
            loss_p6_CA = self.dis_P6_CA(f['p6'], 0.0, _lambdas_CA['p6'],m['p6'], domain='source')     #not p6 
            loss_p5_CA = self.dis_P5_CA(f['p5'], 0.0, _lambdas_CA['p5'],m['p5'], domain='source') 
            loss_p4_CA = self.dis_P4_CA(f['p4'], 0.0, _lambdas_CA['p4'],m['p4'], domain='source') 
            loss_p3_CA = self.dis_P3_CA(f['p3'], 0.0, _lambdas_CA['p3'],m['p3'], domain='source')
#             for name, layer in self.dis_P3.named_modules():
#                 if isinstance(layer, nn.Conv2d):
#                     if '.0' in name:
#                         print('for source',name, layer.weight.sum())
            
        #print('feature shape fp7 ', f['p7'].shape)
        #proposals, proposal_losses = self.proposal_generator(images, f, gt_instances)
        
        proposal_losses["loss_p3"] = loss_p3
        proposal_losses["loss_p4"] = loss_p4
        proposal_losses["loss_p5"] = loss_p5
        proposal_losses["loss_p6"] = loss_p6    #not p6 
        #proposal_losses["loss_p7"] = loss_p7     #not p7 
        
#         #CA proposals
        proposal_losses["loss_p3_CA"] = loss_p3_CA
        proposal_losses["loss_p4_CA"] = loss_p4_CA
        proposal_losses["loss_p5_CA"] = loss_p5_CA
        proposal_losses["loss_p6_CA"] = loss_p6_CA    #not p6 
        #proposal_losses["loss_p7_CA"] = loss_p7_CA     #not p7 

        return proposals, proposal_losses  
    
    def forward(self, batched_inputs, _lambdas={}, _lambdas_CA={} , domain_target = False ):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if len(_lambdas)==0:
            _lambdas['p3']=0.01 #0.5
            _lambdas['p4']=0.01#0.5
            _lambdas['p5']=0.01 #0.5
            _lambdas['p6']=0.01#0.1    #not p6 
            #_lambdas['p7']=0.01#0.1       #not p7
            
            #CA lambdas #skipped length checking
            _lambdas_CA['p3']=0.02 #0.5
            _lambdas_CA['p4']=0.02#0.5
            _lambdas_CA['p5']=0.02 #0.5
            _lambdas_CA['p6']=0.02#0.1    #not p6 
            #_lambdas_CA['p7']=0.02#0.1       #not p7
            
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        #print('input shape', images.tensor.shape)
        features = self.backbone(images.tensor)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        #print("interanl lambdas", _lambdas)
        proposals, proposal_losses = self.losses(images, features, gt_instances , _lambdas, _lambdas_CA, domain_target )
        
#         if domain_target:
#             print('target image', images.tensor.shape, images.tensor.sum())
#             print('for target p3 and p7 sum', features['p3'].sum(), features['p7'].sum() )
#             print('len of proposals for target', len(proposals))
            
#         if not domain_target:
#             print('source image', images.tensor.shape, images.tensor.sum())
#             print('for source p3 and p7 sum', features['p3'].sum() , features['p7'].sum() )
#             print('len of proposals for sourec', len(proposals))

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        
        #proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        #print(proposals)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses
        #print("\n running after training:")
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

   


