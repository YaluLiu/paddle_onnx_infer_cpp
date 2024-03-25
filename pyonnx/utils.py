import os
import glob
import json
import time
import numpy as np
import cv2

def read_json(json_path):
  with open(json_path,"r") as f:
    data = json.load(f)
  return data

def save_json(data,json_path):
  with open(json_path,"w") as f:
    json.dump(data,f)
    
def get_test_images(infer_dir, infer_img):
  """
  Get image path list in TEST mode
  """
  assert infer_img is not None or infer_dir is not None, \
    "--infer_img or --infer_dir should be set"
  assert infer_img is None or os.path.isfile(infer_img), \
      "{} is not a file".format(infer_img)
  assert infer_dir is None or os.path.isdir(infer_dir), \
      "{} is not a directory".format(infer_dir)

  # infer_img has a higher priority
  if infer_img and os.path.isfile(infer_img):
    return [infer_img]

  images = set()
  infer_dir = os.path.abspath(infer_dir)
  assert os.path.isdir(infer_dir), \
    "infer_dir {} is not a directory".format(infer_dir)
  exts = ['jpg', 'jpeg', 'png', 'bmp']
  exts += [ext.upper() for ext in exts]
  for ext in exts:
    images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
  images = list(images)

  assert len(images) > 0, "no image found in {}".format(infer_dir)
  print("Found {} inference images in total.".format(len(images)))

  return images

# {'id': 1, 'width': 2304, 'height': 1296, 'file_name': '.jpg'}
def read_images_from_gt(gt_file_path):
  gt_data = json.load(open(gt_file_path,"r"))
  return gt_data["images"]

# coco格式
# all_labels = {
#   0:  'person',
#   1:  'bicycle',
#   2:  'car',
#   3:  'motorcycle',
#   5:  'bus',
#   7:  "truck",
#   15: 'cat',
#   16: 'dog'
# }

# 标注格式
# 人 1 车2 自行车3 电瓶车4 宠物5
def trans_coco_cls_id(class_id):
  if class_id == 0: # 人
    return 1
  
  if class_id == 2 or class_id == 5 or class_id == 7:
    return 2
  
  if class_id == 1:
    return 3
  
  if class_id == 3:
    return 4
  
  if class_id == 15 or class_id == 16:
    return 5
  
  return -1

def trans_my_own_id(class_id):
  if class_id == 1: # 人
    return 0
  
  if class_id == 2:
    return 2
  
  if class_id == 3:
    return 1
  
  if class_id == 4:
    return 3
  
  if class_id == 5:
    return 15
  
  return -1
  

def make_fake_anno(json_results):
  with open("instances_val2014.json","r") as f:
    instances_example_json = json.load(f)
  
  gt_json = {"images":[],"annotations":[],"categories":instances_example_json["categories"]}

  image_id = 0
  anno_id = 0
  image_prev_name = None
  for x in json_results:
    image_now_name =  x["image_name"]
    if image_prev_name != image_now_name:
      image_id+=1
      image_prev_name = image_now_name
      flag_save_image_json = True
    else:
      flag_save_image_json = False
    
    
    image_json={
      "id":image_id,
      "image_path":x["image_name"]
    }
    x1,y1,x2,y2 = x["bbox"]
    annotation_json = {
      "image_id":image_id,
      "bbox": x["bbox"],
      "category_id": x["category_id"],
      "score": x["score"],
      "id":anno_id,
      "iscrowd":False,
      "segmentation": [[]],
      "area":(x2-x1)*(y2-y1)
      }
    if flag_save_image_json:
      gt_json["images"].append(image_json)
      gt_json["annotations"].append(annotation_json)
      anno_id+=1
  json.dump(gt_json,open("val.json",'w'))


class BenchMark():
  def __init__(self, batch_size,model_type):
    self.dt_json_path = f"dataset/annotations/dt_{model_type}.json"
    self.perf_json_path = f"dataset/annotations/perf_{model_type}.json"
    self.dt_box_list = []
    self.perf_info = {
      "prev":0,
      "infer":0,
      "post":0,
    }
    self.start_time = 0
  
  def start(self):
    self.start_time = time.time()

  def end(self,type):
    cost = time.time() - self.start_time
    self.perf_info[type] += cost

  def update_dt_anno(self,image_id,single_box):
    class_id, score, x1, y1, w, h = single_box
    class_id = trans_coco_cls_id(class_id)
    if class_id == -1:
      return 
    single_result = {
        "image_id":image_id,
        "bbox":[float(x1),float(y1),float(w),float(h)], # can be float
        "category_id":int(class_id),
        "score":round(float(score),2),
    }
    self.dt_box_list.append(single_result)
  

  def save(self):
    save_json(self.dt_box_list, self.dt_json_path)
    with open(self.perf_json_path,"w") as f:
      json.dump(self.perf_info,f,indent=2)


class CocoWorker():
  def __init__(self):
    # # Load the class names from the COCO dataset
    self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    # Generate a color palette for the classes
    self.color_palette = np.random.uniform(0, 255, size=(80, 3))


  def draw_detections(self, img, single_box):
      """
      Draws bounding boxes and labels on the input image based on the detected objects.

      Args:
          img: The input image to draw detections on.
          single_box: class_id, score, x1, y1, w, h = single_box

      Returns:
          None
      """

      # Extract the coordinates of the bounding box
      class_id, score, x1, y1, w, h = single_box
      class_id = int(class_id)
      score = round(float(score),2)
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

if __name__ == "__main__":
  read_images_from_gt("dataset/annotations/instances_default.json")