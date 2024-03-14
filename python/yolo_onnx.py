# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
import torch
import onnxruntime as ort
import time

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

        # # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(80, 3))
        

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


    def read_input_list(self):
        origins = []
        inputs = []
        for image_path in self.input_paths:
          image,origin_image = self.preprocess(image_path)
          inputs.append(image)
          origins.append(origin_image)
        return inputs,origins


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
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        image_data = image_data.astype(np.float32)

        # Return the preprocessed image data
        return image_data,origin_image

    def postprocess(self, input_image, output):
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

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        # print(f"{self.img_width},{self.input_width},{self.img_height},{self.input_height},")
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
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

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def warmup(self,session):
        batch_size = 1 
        img_data = np.random.rand(batch_size, 3, 640, 640).astype(np.float32)
        inputs_dict = {
            'images': img_data,
            # 'im_shape': np.array([[self.input_width,self.input_height]],np.float32),
            # 'scale_factor':np.array([[x_factor,y_factor]],np.float32),
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
        print("input_shape:",input_shape)

        if isinstance(input_shape[2],str):
          self.input_width = 640
          self.input_height = 640
        else:
          self.input_width = input_shape[2]
          self.input_height = input_shape[3]

        # Preprocess the image data
        imgs,origins = self.read_input_list()
        img_data = np.array(imgs)
        
        inputs_dict = {
            'images': img_data
        }

        inputs_name = [a.name for a in session.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}

        # Run inference using the preprocessed image data
        start = time.time()
        outputs = session.run(None, net_inputs)[0]
        cost = time.time() - start
        print(f"solve {img_data.shape[0]} images,cost {cost}s.")
        # print("result:",len(outputs), outputs[0].shape)

        self.visualize(outputs,origins)

    def visualize(self, outputs, origins):
      # Perform post-processing on the outputs to obtain output image.
      for i in range(len(origins)):
        # Get the height and width of the input image
        self.img_height, self.img_width = origins[i].shape[:2]
        vis_image = self.postprocess(origins[i], outputs[i])  # output image
        cv2.imwrite(f"{args.result_dir}/{i}.jpg",vis_image)


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    model_name="yolov8n.onnx"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=f"models/{model_name}", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default="/images/bus.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--source_dir", type=str, default=f"images", help="input dir")
    parser.add_argument("--result_dir", type=str, default="result", help="visualize result dir")
    args = parser.parse_args()

    inputs = ["images/bus.jpg",
              "images/640bus.jpeg",
              "images/dog_0.jpg",
              "images/frame.jpg"]
    detection = YOLOv8(args.model, inputs, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    detection.main()
