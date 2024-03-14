# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time
from utils import get_test_images

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, input_images, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_paths = input_images
        self.inputs=[]
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_height = self.input_width = 0
        self.img_height = self.img_width = 0

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        

    def draw_detections(self, img, box, score, class_id):
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
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

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

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        output = np.array(output)
        expect_boxes = (output[:, 0] > -1) & (output[:, 1] > 0.5)
        np_boxes = output[expect_boxes, :]

        color_list = self.get_color_map_list(80)
        clsid2color = {}
        srcimg = input_image

        for i in range(np_boxes.shape[0]):
            classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
            xmin, ymin, xmax, ymax = int(np_boxes[i, 2]), int(np_boxes[i, 3]), int(np_boxes[i, 4]), int(np_boxes[i, 5])

            if classid not in clsid2color:
                clsid2color[classid] = color_list[classid]
            color = tuple(clsid2color[classid])

            cv2.rectangle(
                srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)

            cv2.putText(
                srcimg,
                self.classes[classid] + ':' + str(round(conf, 3)), (xmin,
                                                                    ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0),
                thickness=2)
        return srcimg

    def warmup(self,session):
        batch_size = 1 
        img_data = np.random.rand(batch_size, 3, 640, 640).astype(np.float32)
        x_factor = 1.0
        y_factor = 1.0
        inputs_dict = {
            'image': img_data,
            'im_shape': np.array([[self.input_width,self.input_height]],np.float32),
            'scale_factor':np.array([[x_factor,y_factor]],np.float32),
        }

        inputs_name = [a.name for a in session.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        outputs = session.run(None, net_inputs)

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        # "CUDAExecutionProvider"
        # "CPUExecutionProvider"
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider"])
        self.warmup(session)
        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape

        if isinstance(input_shape[2],str):
          self.input_width = 640
          self.input_height = 640
        else:
          self.input_width = input_shape[2]
          self.input_height = input_shape[3]



        # Preprocess the image data
        perf_info = {
          "inputs_num":len(self.input_paths),
          "resize":0,
          "infer":0,
        }

        for img_path in self.input_paths:
          start = time.time()
          input_image = self.preprocess(img_path)
          perf_info["resize"] += time.time() - start
          start = time.time()

          x_factor = self.input_width /  self.img_width
          y_factor = self.input_height / self.img_height
          inputs_dict = {
              'image': input_image,
              'im_shape': np.array([[self.input_width,self.input_height]],np.float32),
              'scale_factor':np.array([[y_factor,x_factor]],np.float32),
          }

          inputs_name = [a.name for a in session.get_inputs()]
          net_inputs = {k: inputs_dict[k] for k in inputs_name}

          # Run inference using the preprocessed image data
          
          outputs = session.run(None, net_inputs)
          perf_info["infer"] += time.time() - start

          outs = np.array(outputs[0])
          expect_boxes = (outs[:, 0] > -1) & (outs[:, 1] > 0.5)
          np_boxes = outs[expect_boxes, :]

          color_list = self.get_color_map_list(80)
          clsid2color = {}

          srcimg = cv2.imread(img_path)

          for i in range(np_boxes.shape[0]):
              classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
              xmin, ymin, xmax, ymax = int(np_boxes[i, 2]), int(np_boxes[
                  i, 3]), int(np_boxes[i, 4]), int(np_boxes[i, 5])

              if classid not in clsid2color:
                  clsid2color[classid] = color_list[classid]
              color = tuple(clsid2color[classid])

              cv2.rectangle(
                  srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)

              cv2.putText(
                  srcimg,
                  self.classes[classid] + ':' + str(round(conf, 3)), (xmin,
                                                                      ymin - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.8, (0, 255, 0),
                  thickness=2)

          img_name = img_path.split("/")[-1]
          cv2.imwrite(f"result/{img_name}",srcimg)

        print(f"paddle_onnx:",perf_info)
        # Perform post-processing on the outputs to obtain output image.

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    model_name="yolov8_n_500e_coco.onnx"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=f"models/{model_name}", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default=None, help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--source_dir", type=str, default=f"images", help="input dir")
    parser.add_argument("--result_dir", type=str, default="result", help="visualize result dir")
    args = parser.parse_args()

    inputs = get_test_images(args.source_dir,args.img)
    detection = YOLOv8(args.model, inputs, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    detection.main()
