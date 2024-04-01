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

from pyonnx.utils import read_images_from_gt,BenchMark,CocoWorker


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
        self.onnx_model = f"models/{args.model}"
        self.image_info_list = read_images_from_gt(args.gt_json_path)
        self.bench_mark = BenchMark(args.gt_json_path,"yolo_onnx")

        self.confidence_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.batch_size = args.batch_size

        self.create_session()
        self.coco_worker = CocoWorker()
    
    def create_session(self):
        # Create an inference session using the ONNX model and specify execution providers
        # "CUDAExecutionProvider"
        # "CPUExecutionProvider"
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

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_height, self.input_width))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        image_data = image_data.astype(np.float32)

        # Return the preprocessed image data
        return image_data,origin_image

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        # print(f"{self.img_width},{self.input_width},{self.img_height},{self.input_height},")
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        classes_scores = outputs[:,4:]
        max_score = np.amax(classes_scores,axis = 1)
        max_score_mask = max_score > self.confidence_thres
        outputs = outputs[max_score_mask]

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        net_output=[]
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            x,y,w,h = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            net_output.append([class_id,score,x,y,w,h])
        net_output = np.array(net_output)
        return net_output

    def warmup(self):
        batch_size = 1 
        img_data = np.random.rand(batch_size, 3, 640, 640).astype(np.float32)
        inputs_dict = {
            'images': img_data,
        }

        inputs_name = [a.name for a in self.session.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        outputs = self.session.run(None, net_inputs)


    def main(self):
        self.bench_mark.start()
        batches,origins = self.read_input_list()
        self.bench_mark.end("prev")

        # batch_num*batch_size
        batch_num = len(batches)
        for batch_id in range(batch_num):
            batch_images = batches[batch_id]
            self.bench_mark.start()
            outputs = self.solve_batch(batch_images)
            self.bench_mark.end("infer")

            for i in range(len(batch_images)):
              # Get the height and width of the input image
              origin_image = origins[batch_id][i]
              self.img_height, self.img_width = origin_image.shape[:2]

              self.bench_mark.start()
              net_outputs = self.postprocess(outputs[i])  # output image
              self.bench_mark.end("post")

              # 获取文件名
              image_info = self.image_info_list[batch_id*self.batch_size+i]
              image_name = image_info["file_name"]
              for single_box in net_outputs:
                self.coco_worker.draw_detections(origin_image, single_box)
                self.bench_mark.update_dt_anno(image_info["id"],single_box)

              cv2.imwrite(f"dataset/result/{image_name}",origin_image)
        self.bench_mark.save()



    def solve_batch(self,batch):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        img_data = np.array(batch)
        
        inputs_dict = {
            'images': img_data
        }

        inputs_name = [a.name for a in self.session.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, net_inputs)[0]
        return outputs

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
