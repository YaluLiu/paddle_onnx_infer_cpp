# Ultralytics YOLO 🚀, AGPL-3.0 license
import os, sys
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(".")
if parent_path not in sys.path:
    sys.path.append(parent_path)
    
import argparse

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from pyonnx.utils import read_images_from_gt,BenchMark,CocoWorker
import json
from ultralytics import YOLO 

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, args):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.args = args
        model_path = f"models/{args.model}"
        self.model = YOLO(model_path,task="detect")

        self.image_info_list = read_images_from_gt(args.gt_json_path)

        suffix = args.model.split(".")[1]
        if suffix == "pt":
          model_type = "ori"
        elif suffix == "engine":
          model_type = "trt"
        elif suffix == "onnx":
          model_type = "onnx"

        self.bench_mark = BenchMark(args.gt_json_path,f"yolo_{model_type}")

        self.confidence_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.batch_size = args.batch_size

        self.input_height = 640
        self.input_width = 640
        self.warmup()
        self.coco_worker = CocoWorker()

    def read_input_list(self):
        origins = []
        inputs = []
        for image_info in self.image_info_list:
          image_name = image_info["file_name"]
          image_path = f"dataset/images/{image_name}"
          image,origin_image = self.preprocess(image_path)
          inputs.append(image)
          origins.append(origin_image)
        
        input_batches = [inputs[i:i+self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        origin_batches = [origins[i:i+self.batch_size] for i in range(0, len(origins), self.batch_size)]
        return input_batches,origin_batches


    def preprocess(self,input_path):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        origin_image = cv2.imread(input_path)

        # image_data = cv2.resize(origin_image, (640,640))

        # Return the preprocessed image data
        return origin_image,origin_image

    def warmup(self):
        warm_input = np.random.rand(self.input_height,self.input_width,3).astype(np.uint8)
        self.model.predict(warm_input, save=False, imgsz=(self.input_height,self.input_width), device=0,verbose=False)

    def main(self):
        self.bench_mark.start()
        batches,origins = self.read_input_list()
        self.bench_mark.end("prev")

        # batch_num*batch_size
        batch_num = len(batches)
        for batch_id in range(batch_num):
            batch_images = batches[batch_id]
            self.bench_mark.start()
            net_outputs = self.solve_batch(batch_images)
            self.bench_mark.end("infer")

            for i in range(len(batch_images)):
              # 获取文件名
              image_info = self.image_info_list[batch_id*self.batch_size+i]
              image_name = image_info["file_name"]
              origin_image = origins[batch_id][i]

              net_output = net_outputs[i]
              boxes = net_output.boxes
              for box in boxes:
                x,y,x2,y2 = box.xyxy[0]
                w = x2 - x
                h = y2 - y
                cls = box.cls 
                conf = box.conf
                single_box = [cls,conf,x,y,w,h]
                self.coco_worker.draw_detections(origin_image,single_box)
                self.bench_mark.update_dt_anno(image_info["id"],single_box)

              cv2.imwrite(f"dataset/result/{image_name}",origin_image)
        self.bench_mark.save()



    def solve_batch(self,batch_images):
        net_outputs = self.model.predict(batch_images, save=False, imgsz=(self.input_height,self.input_width), device=0,verbose=False)
        return net_outputs

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size for model input")

    parser.add_argument("--img", type=str, default=None, help="Path to input image.")
    parser.add_argument("--source_dir", type=str, default=f"images", help="input dir")
    parser.add_argument("--result_dir", type=str, default="result", help="visualize result dir")
    parser.add_argument("--gt_json_path", type=str, default="dataset/annotations/instances_default.json", help="input dir")
    
    args = parser.parse_args()

    
    detection = YOLOv8(args)

    # Perform object detection and obtain the output image
    detection.main()
