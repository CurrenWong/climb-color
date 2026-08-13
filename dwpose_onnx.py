"""DWPose ONNX 推理模块（从 comfyui_controlnet_aux 提取，无 torch 依赖）。

133 点 COCO-WholeBody 索引：
  0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
  11-12 hips, 13-14 knees, 15-16 ankles,
  17 left_big_toe, 18 left_small_toe, 19 left_heel,
  20 right_big_toe, 21 right_small_toe, 22 right_heel,
  23-90 face (68), 91-111 left hand (21), 112-132 right hand (21)
"""
import numpy as np
import cv2
import onnxruntime as ort

# ---- 常量 ----
MODEL_INPUT_SIZE = (288, 384)   # (w, h)
SIMCC_SPLIT_RATIO = 2.0
DET_INPUT_SIZE = (640, 640)
DET_NMS_THR = 0.45
DET_SCORE_THR = 0.1
DET_FINAL_THR = 0.3

# 133 点索引
IDX_L_WRIST = 9
IDX_R_WRIST = 10
IDX_L_ANKLE = 15
IDX_R_ANKLE = 16
IDX_L_BIG_TOE = 17
IDX_L_SMALL_TOE = 18
IDX_L_HEEL = 19
IDX_R_BIG_TOE = 20
IDX_R_SMALL_TOE = 21
IDX_R_HEEL = 22


class DWPoseONNX:
    def __init__(self, det_path, pose_path):
        self.det_sess = ort.InferenceSession(det_path, providers=['CPUExecutionProvider'])
        self.pose_sess = ort.InferenceSession(pose_path, providers=['CPUExecutionProvider'])

    # ---------- 检测器（YOLOX） ----------
    def detect(self, oriImg):
        """返回人体框 (N,4) xyxy，按最大框优先（取最大面积的人）"""
        input_shape = DET_INPUT_SIZE
        img, ratio = self._det_preprocess(oriImg, input_shape)
        input = img[None, :, :, :].astype(np.float32)
        input_name = self.det_sess.get_inputs()[0].name
        output = self.det_sess.run(None, {input_name: input})[0]

        predictions = self._demo_postprocess(output, input_shape)[0]
        boxes = predictions[:, :4]
        # 该 onnx 输出为 logits（obj/cls 需 sigmoid 才是概率）
        obj = 1.0 / (1.0 + np.exp(-predictions[:, 4]))
        cls = 1.0 / (1.0 + np.exp(-predictions[:, 5:]))
        scores = obj[:, None] * cls
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        dets = self._multiclass_nms(boxes_xyxy, scores, DET_NMS_THR, DET_SCORE_THR)
        if dets is None:
            return np.empty((0, 4), dtype=np.float32)
        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls_inds = dets[:, 5]
        isscore = final_scores > DET_FINAL_THR
        iscat = final_cls_inds == 0  # person
        mask = isscore & iscat
        boxes = final_boxes[mask]
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        # 按置信度排序，取最高分的人（score 比面积更可靠）
        boxes = boxes[np.argsort(-final_scores[mask])]
        return boxes

    @staticmethod
    def _det_preprocess(img, input_size):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                 interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    @staticmethod
    def _demo_postprocess(outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [img_size[0] // s for s in strides]
        wsizes = [img_size[1] // s for s in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    @staticmethod
    def _nms(boxes, scores, nms_thr):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep

    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid = cls_scores > score_thr
            if valid.sum() == 0:
                continue
            vscores = cls_scores[valid]
            vboxes = boxes[valid]
            keep = self._nms(vboxes, vscores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([vboxes[keep], vscores[keep, None], cls_inds], 1)
                final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    # ---------- 姿态估计（DWPose RTMPose） ----------
    def pose(self, oriImg, bboxes):
        """对每张 bbox 做 top-down 姿态估计。

        返回 (keypoints, scores):
          keypoints: (N, 133, 2) 原图坐标
          scores:    (N, 133)
        """
        if len(bboxes) == 0:
            bboxes = np.array([[0, 0, oriImg.shape[1], oriImg.shape[0]]], dtype=np.float32)
        imgs, centers, scales = self._pose_preprocess(oriImg, bboxes)
        outputs = self._pose_inference(imgs)
        keypoints, scores = self._pose_postprocess(outputs, centers, scales)
        return keypoints, scores

    def _pose_preprocess(self, img, out_bbox):
        img_shape = img.shape[:2]
        out_img, out_center, out_scale = [], [], []
        for i in range(len(out_bbox)):
            x0, y0, x1, y1 = out_bbox[i]
            bbox = np.array([x0, y0, x1, y1])
            center, scale = self._bbox_xyxy2cs(bbox, padding=1.25)
            resized_img, scale = self._top_down_affine(MODEL_INPUT_SIZE, scale, center, img)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            resized_img = (resized_img - mean) / std
            out_img.append(resized_img)
            out_center.append(center)
            out_scale.append(scale)
        return out_img, out_center, out_scale

    def _pose_inference(self, imgs):
        input = np.stack(imgs, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        input_name = self.pose_sess.get_inputs()[0].name
        all_outputs = self.pose_sess.run(None, {input_name: input})
        # 每张图的 [simcc_x, simcc_y]
        outs = []
        for b in range(input.shape[0]):
            outs.append([all_outputs[0][b:b+1], all_outputs[1][b:b+1]])
        return outs

    def _pose_postprocess(self, outputs, center, scale):
        all_key, all_score = [], []
        for i in range(len(outputs)):
            simcc_x, simcc_y = outputs[i]
            keypoints, scores = self._decode(simcc_x, simcc_y, SIMCC_SPLIT_RATIO)
            keypoints = keypoints / MODEL_INPUT_SIZE * scale[i] + center[i] - scale[i] / 2
            all_key.append(keypoints[0])
            all_score.append(scores[0])
        return np.array(all_key), np.array(all_score)

    @staticmethod
    def _bbox_xyxy2cs(bbox, padding=1.0):
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding
        if dim == 1:
            center = center[0]
            scale = scale[0]
        return center, scale

    @staticmethod
    def _fix_aspect_ratio(bbox_scale, aspect_ratio):
        w, h = np.hsplit(bbox_scale, [1])
        return np.where(w > h * aspect_ratio,
                        np.hstack([w, w / aspect_ratio]),
                        np.hstack([h * aspect_ratio, h]))

    @staticmethod
    def _rotate_point(pt, angle_rad):
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    @staticmethod
    def _get_3rd_point(a, b):
        direction = a - b
        return b + np.r_[-direction[1], direction[0]]

    def _get_warp_matrix(self, center, scale, rot, output_size, shift=(0., 0.), inv=False):
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])
        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat

    def _top_down_affine(self, input_size, bbox_scale, bbox_center, img):
        w, h = input_size
        warp_size = (int(w), int(h))
        bbox_scale = self._fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
        rot = 0
        warp_mat = self._get_warp_matrix(bbox_center, bbox_scale, rot, output_size=(w, h))
        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
        return img, bbox_scale

    @staticmethod
    def _get_simcc_maximum(simcc_x, simcc_y):
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)
        return locs, vals

    def _decode(self, simcc_x, simcc_y, simcc_split_ratio):
        keypoints, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio
        return keypoints, scores

    # ---------- 高层封装 ----------
    def run(self, oriImg):
        """整图推理：返回 133 点 keypoints (N,133,2) + scores (N,133)，取最大的人"""
        boxes = self.detect(oriImg)
        if len(boxes) == 0:
            return None, None, None
        keypoints, scores = self.pose(oriImg, boxes[:1])  # 只取最大的人
        return boxes[:1], keypoints, scores

    def extract(self, kps, scores):
        """从 133 点提取攀岩端点。返回 dict，每个值为 (x,y) 或 None（置信度<0.3 视为缺失）"""
        thr = 0.3
        def pt(idx):
            if scores[idx] < thr:
                return None
            return (float(kps[idx, 0]), float(kps[idx, 1]))
        return {
            'L_hand': pt(IDX_L_WRIST),
            'R_hand': pt(IDX_R_WRIST),
            'L_foot': pt(IDX_L_BIG_TOE),   # 用脚尖
            'R_foot': pt(IDX_R_BIG_TOE),
            'L_ankle': pt(IDX_L_ANKLE),
            'R_ankle': pt(IDX_R_ANKLE),
        }
