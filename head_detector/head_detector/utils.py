import torch
import torchvision


CONFIDENCE_THRESHOLD = 0.5


def nms(boxes_xyxy, scores, flame_params, threshold: float = 0.5, top_k: int = 1000, keep_top_k: int = 100):
    for pred_bboxes_xyxy, pred_bboxes_conf, pred_flame_params in zip(
            boxes_xyxy.detach().float(),
            scores.detach().float(),
            flame_params.detach().float(),
    ):
        pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
        conf_mask = pred_bboxes_conf >= CONFIDENCE_THRESHOLD

        pred_bboxes_conf = pred_bboxes_conf[conf_mask]
        pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask]
        pred_flame_params = pred_flame_params[conf_mask]

        # Filter all predictions by self.nms_top_k
        if pred_bboxes_conf.size(0) > top_k:
            topk_candidates = torch.topk(pred_bboxes_conf, k=top_k, largest=True, sorted=True)
            pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
            pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
            pred_flame_params = pred_flame_params[topk_candidates.indices]

        # NMS
        idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf,
                                                iou_threshold=threshold)

        final_bboxes = pred_bboxes_xyxy[idx_to_keep][: keep_top_k]  # [Instances, 4]
        final_scores = pred_bboxes_conf[idx_to_keep][: keep_top_k]  # [Instances, 1]
        final_params = pred_flame_params[idx_to_keep][: keep_top_k]  # [Instances, Flame Params]
        return final_bboxes, final_scores, final_params