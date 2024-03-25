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
from pyonnx.utils import BenchMark,read_images_from_gt,CocoWorker

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
        self.onnx_model = f"models/{args.model}"
        self.image_info_list = read_images_from_gt(args.gt_json_path)
        self.confidence_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.input_height = self.input_width = 0
        self.img_height = self.img_width = 0
        self.batch_size = 1
        self.bench_mark = BenchMark(self.batch_size,"paddle_onnx")
        self.coco_worker = CocoWorker()
        self.create_session()


    def preprocess(self,input_path):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(input_path)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def warmup(self):
        img_data = np.random.rand(self.batch_size, 3, 640, 640).astype(np.float32)
        x_factor = 1.0
        y_factor = 1.0
        inputs_dict = {
            'image': img_data,
            'im_shape': np.array([[self.input_width,self.input_height]],np.float32),
            'scale_factor':np.array([[x_factor,y_factor]],np.float32),
        }

        inputs_name = [a.name for a in self.session.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        outputs = self.session.run(None, net_inputs)

    def create_session(self):
        self.session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.warmup()
        # Get the model inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape

        if isinstance(input_shape[2],str):
          self.input_width = 640
          self.input_height = 640
        else:
          self.input_width = input_shape[2]
          self.input_height = input_shape[3]

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        for image_info in self.image_info_list:
          image_name = image_info["file_name"]
          image_path = f"dataset/images/{image_name}"
          image_id = image_info["id"] 

          self.bench_mark.start()
          image = self.preprocess(image_path)
          self.bench_mark.end("prev")

          self.bench_mark.start()
          x_factor = self.input_width /  self.img_width
          y_factor = self.input_height / self.img_height
          inputs_dict = {
              'image': image,
              'im_shape': np.array([[self.input_width,self.input_height]],np.float32),
              'scale_factor':np.array([[y_factor,x_factor]],np.float32),
          }

          inputs_name = [a.name for a in self.session.get_inputs()]
          net_inputs = {k: inputs_dict[k] for k in inputs_name}

          # Run inference using the preprocessed image data
          outputs = self.session.run(None, net_inputs)
          self.bench_mark.end("infer")

          outs = np.array(outputs[0])
          expect_boxes = (outs[:, 0] > -1) & (outs[:, 1] > self.confidence_thres)
          np_boxes = outs[expect_boxes, :]


          srcimg = cv2.imread(image_path)
          for i in range(np_boxes.shape[0]):
              classid, conf,x0,y0,x1,y1 = np_boxes[i]
              w = x1-x0
              h = y1-y0
              single_box = [classid,conf,x0,y0,w,h]
              self.coco_worker.draw_detections(srcimg,single_box)
              self.bench_mark.update_dt_anno(image_id,single_box)
          
          img_name = image_path.split("/")[-1]
          cv2.imwrite(f"dataset/result/{img_name}",srcimg)

        self.bench_mark.save()

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    model_name="yolov8_n_500e_coco.onnx"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=f"{model_name}", help="Input your ONNX model.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")

    parser.add_argument("--img", type=str, default=None, help="Path to input image.")
    parser.add_argument("--gt_json_path", type=str, default="dataset/annotations/instances_default.json", help="input dir")
    parser.add_argument("--source_dir", type=str, default=f"images", help="input dir")
    parser.add_argument("--result_dir", type=str, default="dataset/result", help="visualize result dir")

    args = parser.parse_args()
    detection = YOLOv8(args)

    # Perform object detection and obtain the output image
    detection.main()
