import json
from utils import trans_my_own_id,read_json,save_json
import cv2
import numpy as np 
from ultralytics.utils import  yaml_load
from ultralytics.utils.checks import check_yaml

# # Load the class names from the COCO dataset
classes = yaml_load(check_yaml("coco128.yaml"))["names"]

# Generate a color palette for the classes
color_palette = np.random.uniform(0, 255, size=(80, 3))
  

def merge_box(gt_json):
  image_info_list = gt_json["images"]
  box_data_list = gt_json["annotations"]
  merge_list = {}
  #添加图片路径
  for image_info in image_info_list:
    image_id = image_info["id"]
    if image_id not in merge_list.keys():
      merge_list[image_id] = {}
      merge_list[image_id]["anno"] = []
      merge_list[image_id]["file_name"] = image_info["file_name"]
    else:
      print("Error")

  # 添加图片标注信息
  for box_data in box_data_list:
    image_id = box_data["image_id"]
    merge_list[image_id]["anno"].append(box_data)
  return merge_list

def draw_detections(img, single_box):
    # Extract the coordinates of the bounding box
    class_id, score, x1, y1, w, h = single_box
    class_id = int(class_id)
    x1 = int(x1)
    y1 = int(y1)
    w = int(w)
    h = int(h)

    # Retrieve the color for the class ID
    color = color_palette[class_id]

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 2)

    # Create the label text with class name and score
    label = f"{classes[class_id]}: {score:.2f}"

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


def viusualize_image(merge_list):
  image_ids = merge_list.keys()
  for image_id in image_ids:
    image_info = merge_list[image_id]
    if "file_name" not in image_info.keys():
      continue
    image_data = image_info["anno"]
    image_name = image_info["file_name"]
    image_path = f"{dataset_dir}/images/{image_name}"
    vis_path = f"{dataset_dir}/result/{image_name}"
    img = cv2.imread(image_path)
    for box_data in image_data:
      bbox = box_data["bbox"]
      x1,y1,w,h=bbox
      class_id = box_data["category_id"]
      class_id = trans_my_own_id(class_id)
      score = 1.0
      single_box = [class_id, score, x1, y1, w, h]
      draw_detections(img, single_box)
    cv2.imwrite(vis_path,img)


def load_dt_box(dt_json,merge_list):
  dt_list = read_json(dt_json)
  #清空
  for image_id in merge_list.keys():
    merge_list[image_id]["anno"] = []

  # 重新加载det数据
  for box_data in dt_list:
    image_id = box_data["image_id"]
    merge_list[image_id]["anno"].append(box_data)
  return merge_list


def save_gt_to_dt(gt_data,json_path):
  dt = gt_data["annotations"]
  for data in dt:
    data["score"] = 0.9
  save_json(dt,json_path)



if __name__ == "__main__":
  dataset_dir="dataset"
  gt_name = "instances_default.json"
  # dt_name = "instances_default_dt_yolo.json"
  dt_name = "fake_gt_to_dt.json"
  gt_path = f"{dataset_dir}/annotations/{gt_name}"
  dt_path = f"{dataset_dir}/annotations/{dt_name}"

  gt_json = read_json(gt_path)
  # 利用gt制造假的dt数据
  save_gt_to_dt(gt_json,dt_path)

  # 将gt-json的image和anno合并
  merge_list = merge_box(gt_json)

  # 测试dt数据正确性
  merge_list = load_dt_box(dt_path,merge_list)

  viusualize_image(merge_list)