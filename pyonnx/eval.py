
import re
import os
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict
import argparse
from pathlib import Path
import numpy as np 
 
class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }
 
    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results
 
    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval
 
        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]
 
    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results
 
 
if __name__ == '__main__':
    # instances_val2014.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt',default=".json")
    parser.add_argument('--gt',default=".json")
    parser.add_argument('--image-dir',default="")
    args = parser.parse_args()


    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.dt)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # imgIds=sorted(coco_gt.getImgIds())
    # imgIds=imgIds[0:100]
    # imgIds = [42,73,74,133]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # results = COCOResults('bbox')
    # results.update(coco_eval)
    # print(results)