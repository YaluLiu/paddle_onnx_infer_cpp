import numpy as np
import cv2
import os 
from ultralytics import YOLO 
from timeit import default_timer as timer
from copy import deepcopy
import argparse
import time

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default=f"models/yolov8n", help="Input your model.")
  parser.add_argument("--input_w", type=int, default=640, help="input_w")
  parser.add_argument("--input_h", type=int, default=640, help="input_h")
  parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
  parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
  parser.add_argument("--source_dir", type=str, default=f"images", help="input dir")
  parser.add_argument("--batch_size", type=int, default=10, help="batch_size")
  args = parser.parse_args()
  return args

class YoloSpeed:
  def __init__(self,model_format=".engine"):
    args = read_args()
    self.args = args
    self.model_format = model_format
    model_path = self.args.model+"."+self.model_format
    self.export(model_path)
    self.model = YOLO(model_path,task="detect")
    warm_input = np.random.rand(self.args.input_h,self.args.input_w,3).astype(np.uint8)
    self.model.predict(warm_input, save=False, imgsz=(self.args.input_h,self.args.input_w), device=0,verbose=False)

  def export(self,model_path):
    if os.path.exists(model_path):
      return
    model = YOLO('yolov8n.pt')
    model.export(
        format=self.model_format, 
        imgsz=(self.args.input_h,self.args.input_w),
        half=False,
        dynamic=True,
        simplify=False,
        opset=12, 
        device=0)
    
  def read_source(self):
    image_list = []
    source = os.listdir(self.args.source_dir)
    for image_name in source:
      image_path = os.path.join(self.args.source_dir,image_name)
      if os.path.isfile(image_path):
        image_list.append(image_path)
    return image_list

  
  def preprocess(self, source):
    inputs= [self.read_image(img_name) for img_name in source]
    return inputs
  
  def read_image(self,img_path):
    if img_path.endswith(".data"):
      pass
    else:
      return cv2.imread(img_path)
  
  def predict(self,inputs):
    net_outputs = self.model.predict(inputs, save=False, imgsz=(self.args.input_h,self.args.input_w), device=0,verbose=False)
    return net_outputs
  
  def run(self):
    source = self.read_source()
    all_inputs = self.preprocess(source)

    perf_info = {
      "inputs_num":len(all_inputs),
      "resize":0,
      "infer":0,
    }
    # resize
    start = time.time()
    all_inputs = [cv2.resize(img, (640,640)) for img in all_inputs]
    perf_info["resize"] += time.time()-start

    batch_size = self.args.batch_size
    batches = [all_inputs[i:i+batch_size] for i in range(0, len(all_inputs), batch_size)]
    for batch in batches:
      start = time.time()
      net_outputs = self.predict(batch)
      perf_info["infer"] +=  time.time()-start
    return perf_info
    

if __name__ == "__main__":
  yolo_speed = YoloSpeed("engine")
  tensorrt_perf_info = yolo_speed.run()
  yolo_speed = YoloSpeed("onnx")
  onnx_perf_info = yolo_speed.run()

  print("-------------------------")
  print(f"tensorrt_perf_info:",tensorrt_perf_info)
  print(f"onnx_perf_info:",onnx_perf_info)

  