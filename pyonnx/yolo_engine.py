# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from utils import read_images_from_gt,BenchMark
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
        self.model = YOLO(args.model,task="detect")

        self.image_info_list = read_images_from_gt(args.gt_json_path)
        self.bench_mark = BenchMark(args.gt_json_path,"yolo_trt")

        self.confidence_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.batch_size = args.batch_size

        self.input_height = 640
        self.input_width = 640
        self.warmup()

        # # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(80, 3))

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


    def draw_detections(self, img, single_box):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        class_id, score, x1, y1, w, h = single_box
        score = round(float(score),2)
        class_id = int(class_id)
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


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
                # import pdb
                # pdb.set_trace()
                x,y,x2,y2 = box.xyxy[0]
                w = x2 - x
                h = y2 - y
                cls = box.cls 
                conf = box.conf
                single_box = [cls,conf,x,y,w,h]
                self.draw_detections(origin_image, single_box)
                self.bench_mark.update_dt_anno(image_info["id"],single_box)

              cv2.imwrite(f"dataset/result/{image_name}",origin_image)
        self.bench_mark.save()



    def solve_batch(self,batch_images):
        net_outputs = self.model.predict(batch_images, save=False, imgsz=(self.input_height,self.input_width), device=0,verbose=False)
        return net_outputs

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    model_name="yolov8n.onnx"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=f"models/{model_name}", help="Input your ONNX model.")
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
