# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import numpy as np
import math
import cv2
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..keypoint_utils import transform_preds
from .. import layers as L

__all__ = ['TopDownHRNet', 'HRNetPostProcess', 'TopDownHRNetSimDR', 'SimDRPostProcess']


@register
class TopDownHRNet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 width,
                 num_joints,
                 backbone='HRNet',
                 loss='KeyPointMSELoss',
                 post_process='HRNetPostProcess',
                 flip_perm=None,
                 flip=True,
                 shift_heatmap=True,
                 use_dark=True):
        """
        HRNet network, see https://arxiv.org/abs/1902.09212

        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): `HRNetPostProcess` instance
            flip_perm (list): The left-right joints exchange order list
            use_dark(bool): Whether to use DARK in post processing
        """
        super(TopDownHRNet, self).__init__()
        self.backbone = backbone
        self.post_process = HRNetPostProcess(use_dark)
        self.loss = loss
        self.flip_perm = flip_perm
        self.flip = flip
        self.final_conv = L.Conv2d(width, num_joints, 1, 1, 0, bias=True)
        self.shift_heatmap = shift_heatmap
        self.deploy = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        return {'backbone': backbone, }

    def _forward(self):
        feats = self.backbone(self.inputs)
        hrnet_outputs = self.final_conv(feats[0])

        if self.training:
            return self.loss(hrnet_outputs, self.inputs)
        elif self.deploy:
            outshape = hrnet_outputs.shape
            max_idx = paddle.argmax(
                hrnet_outputs.reshape(
                    (outshape[0], outshape[1], outshape[2] * outshape[3])),
                axis=-1)
            return hrnet_outputs, max_idx
        else:
            if self.flip:
                self.inputs['image'] = self.inputs['image'].flip([3])
                feats = self.backbone(self.inputs)
                output_flipped = self.final_conv(feats[0])
                output_flipped = self.flip_back(output_flipped.numpy(),
                                                self.flip_perm)
                output_flipped = paddle.to_tensor(output_flipped)
                if self.shift_heatmap:
                    output_flipped[:, :, :, 1:] = output_flipped.clone(
                    )[:, :, :, 0:-1]
                hrnet_outputs = (hrnet_outputs + output_flipped) * 0.5
            imshape = (self.inputs['im_shape'].numpy()
                       )[:, ::-1] if 'im_shape' in self.inputs else None
            center = self.inputs['center'].numpy(
            ) if 'center' in self.inputs else np.round(imshape / 2.)
            scale = self.inputs['scale'].numpy(
            ) if 'scale' in self.inputs else imshape / 200.
            outputs = self.post_process(hrnet_outputs, center, scale)
            return outputs

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'keypoint': res_lst}
        return outputs

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped


class HRNetPostProcess(object):
    def __init__(self, use_dark=True):
        self.use_dark = use_dark

    def get_max_preds(self, heatmaps):
        '''get predictions from score maps

        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        '''
        assert isinstance(heatmaps,
                          np.ndarray), 'heatmaps should be numpy.ndarray'
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        idx = idx.reshape((batch_size, num_joints, 1))
        maxvals = np.amax(heatmaps_reshaped, 2, keepdims=True)

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
                + hm[py-1][px-1])
            dyy = 0.25 * (
                hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        '''DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).
        '''

        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """
        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array([
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ])
                        coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i],
                                       [heatmap_width, heatmap_height])

        return preds, maxvals

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(output.numpy(), center, scale)
        outputs = [[
            np.concatenate(
                (preds, maxvals), axis=-1), np.mean(
                    maxvals, axis=1)
        ]]
        return outputs

@register
class TopDownHRNetSimDR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 width,
                 num_joints,
                 shape,
                 backbone='HRNet',
                 loss='KLDiscretLoss',
                 post_process='SimDRPostProcess',
                 flip_perm=None,
                 flip=True,
                 shift_heatmap=False,
                 use_dark=True,
                 split_ratio = 2.0,
                 freeze_final_conv = False):
        """
        HRNet network, see https://arxiv.org/abs/1902.09212

        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): `SimDRPostProcess` instance
            flip_perm (list): The left-right joints exchange order list
            use_dark(bool): Whether to use DARK in post processing
        """
        super(TopDownHRNetSimDR, self).__init__()
        self.backbone = backbone
        self.post_process = SimDRPostProcess(use_dark)
        self.loss = loss
        self.flip_perm = flip_perm
        self.flip = flip
        self.final_conv = L.Conv2d(width, num_joints, 1, 1, 0, bias=True)
        if freeze_final_conv: self.final_conv.stop_gradient = True
        self.shift_heatmap = shift_heatmap
        self.deploy = False
        self.mlp_head_x = nn.Linear(np.product(shape[0]), int(shape[1][0]*split_ratio))
        self.mlp_head_y = nn.Linear(np.product(shape[0]), int(shape[1][1]*split_ratio))
    
    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        return {'backbone': backbone, }

    def _forward(self):
        feats = self.backbone(self.inputs)
        feats = self.final_conv(feats[0])
        B, C, H, W = feats.shape
        feats = feats.reshape((B, C, H*W)) #'b c h w -> b c (h w)'
        pred_x = self.mlp_head_x(feats)
        pred_y = self.mlp_head_y(feats)
        pred = [pred_x, pred_y]
        if self.training:
            return self.loss(pred, self.inputs)
        elif self.deploy:
            max_idx = paddle.argmax(feats, axis=-1)
            return pred, max_idx
        else:
            if self.flip:
                self.inputs['image'] = self.inputs['image'].flip([3])
                feats = self.backbone(self.inputs)
                feats = self.final_conv(feats[0]).reshape((B, C, H*W))
                pred_x = self.mlp_head_x(feats)
                pred_y = self.mlp_head_y(feats)
                del feats
                output_flipped = [pred_x, pred_y]
                output_flipped[0] = self.flip_back(output_flipped[0].numpy(),
                                                self.flip_perm, "x")
                output_flipped[0] = paddle.to_tensor(output_flipped[0])
                output_flipped[1] = self.flip_back(output_flipped[1].numpy(),
                                                self.flip_perm, "y")
                output_flipped[1] = paddle.to_tensor(output_flipped[1])
                if self.shift_heatmap:
                    output_flipped[0][:, :, 1:] = output_flipped[0].clone()[:, :, 0:-1]
                    output_flipped[1][:, :, 1:] = output_flipped[1].clone()[:, :, 0:-1]
                pred[0] = (pred[0] + output_flipped[0]) * 0.5
                pred[1] = (pred[1] + output_flipped[1]) * 0.5
            imshape = (self.inputs['im_shape'].numpy()
                       )[:, ::-1] if 'im_shape' in self.inputs else None
            center = self.inputs['center'].numpy(
            ) if 'center' in self.inputs else np.round(imshape / 2.)
            scale = self.inputs['scale'].numpy(
            ) if 'scale' in self.inputs else imshape / 200.
            outputs = self.post_process(pred, center, scale)
            return outputs

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'keypoint': res_lst}
        return outputs

    def flip_back(self, output_flipped, matched_parts, type='x'):
        assert output_flipped.ndim == 3,\
                'output_flipped should be [batch_size, num_joints, size]'

        if type == 'x':
            output_flipped = output_flipped[:, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :].copy()
            output_flipped[:, pair[0], :] = output_flipped[:, pair[1], :]
            output_flipped[:, pair[1], :] = tmp

        return output_flipped


class SimDRPostProcess(object):
    def __init__(self, use_dark=True):
        self.use_dark = use_dark
        self.postprocess = False

    def get_max_preds(self, heatmaps):
        '''get predictions from score maps

        Args:
            heatmaps: [numpy.ndarray([batch_size, num_joints, width]), numpy.ndarray([batch_size, num_joints, height])]

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        '''
        assert isinstance(heatmaps,
                          list), 'heatmaps should be list of numpy.ndarray'
        assert heatmaps[0].ndim == 3, 'batch_images should be 3-ndim'
        assert heatmaps[1].ndim == 3, 'batch_images should be 3-ndim'

        #batch_size = heatmaps[0].shape[0]
        #num_joints = heatmaps[0].shape[1]
        #width = heatmaps[0].shape[2]
        #height = heatmaps[1].shape[2]
        idx_x = np.argmax(heatmaps[0], 2)
        idx_y = np.argmax(heatmaps[1], 2)
        preds = np.stack([idx_x, idx_y], -1).astype(np.float32)
        
        maxvals_x = np.amax(heatmaps[0], 2, keepdims=True)
        maxvals_y = np.amax(heatmaps[1], 2, keepdims=True)
        maxvals = np.maximum(maxvals_x, maxvals_y)

        return preds, maxvals

    def mask_preds(self, preds, maxvals):
        #pred_mask = np.concatenate([np.greater(maxvals_x, 0.0), np.greater(maxvals_y, 0.0)], -1)
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        preds *= pred_mask
        return preds, maxvals

    def dark_parse(self, hm, coord):
        width = hm.shape[0]
        px = round(coord)
        if 1 < px < width - 2:
            #dx = 0.5 * (hm[px + 1] - hm[px - 1])
            #dxx = 0.25 * (hm[px + 2] - 2 * hm[px] + hm[px - 2])
            #dx = 2 * (hm[px + 1] - hm[px - 1])
            #dxx = (hm[px + 2] - 2 * hm[px] + hm[px - 2])
            #coord -= dx/dxx if dxx !=0 else 0
            delta = 2 * (hm[px + 1] - hm[px - 1])/(hm[px + 2] - 2 * hm[px] + hm[px - 2])
            coord -= delta if np.abs(delta)<1 else 0
        return coord

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        size = heatmap.shape[2]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((size + 2 * border))
                dr[border:-border] = heatmap[i, j]
                dr = cv2.GaussianBlur(dr, (1, kernel), 0).squeeze()
                heatmap[i, j] = dr[border:-border]
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_postprocess(self, hm, coords, kernelsize):
        '''DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).
        
        hm (list): The predicted heatmaps [numpy.ndarray([batch_size, num_joints, width]), numpy.ndarray([batch_size, num_joints, height])]
        coords: numpy.ndarray([batch_size, num_joints, 2])
        '''

        hm = [np.log(np.maximum(self.gaussian_blur(hm[0], kernelsize), 1e-10)), 
                np.log(np.maximum(self.gaussian_blur(hm[1], kernelsize), 1e-10))]
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p, 0] = self.dark_parse(hm[0][n][p], coords[n][p][0])
                coords[n, p, 1] = self.dark_parse(hm[1][n][p], coords[n][p][1])
        return coords

    def get_final_preds(self, output, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (list): The predicted heatmaps [numpy.ndarray([batch_size, num_joints, width]), numpy.ndarray([batch_size, num_joints, height])]
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """
        output = [output[0].numpy(),output[1].numpy()]
        coords, maxvals = self.get_max_preds(output)
        #coords, maxvals = self.mask_preds(coords, maxvals)
        width = output[0].shape[2]
        height = output[1].shape[2]
        if self.postprocess:
            if self.use_dark:
                coords = self.dark_postprocess(output, coords, kernelsize)
            else:
                for n in range(coords.shape[0]):
                    for p in range(coords.shape[1]):
                        hm_x = output[0][n][p]
                        hm_y = output[1][n][p]
                        px = int(math.floor(coords[n][p][0] + 0.5))
                        py = int(math.floor(coords[n][p][1] + 0.5))
                        if 1 < px < width - 1 and 1 < py < height - 1:
                            coords[n][p][0] += np.sign(hm_x[px + 1] - hm_x[px - 1]) * .25
                            coords[n][p][1]  += np.sign(hm_y[py + 1] - hm_y[py - 1]) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i], [width, height])
        return preds, maxvals

    def get_max_preds_tensor(self, output):
        output_x, output_y = output
        max_val_x, coords_x = output_x.max(2, keepdims=True), output_x.argmax(2, keepdim=True)
        max_val_y, coords_y = output_y.max(2, keepdims=True), output_y.argmax(2, keepdim=True)

        maxvals = paddle.maximum(max_val_x, max_val_y).cpu().numpy()

        coords = paddle.concat([coords_x, coords_y], -1).astype('float32').cpu().numpy()
        return  coords, maxvals

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(output, center, scale)

        outputs = [[
            np.concatenate((preds, maxvals), axis=-1),
            np.mean(maxvals, axis=1)
        ]]
        return outputs

