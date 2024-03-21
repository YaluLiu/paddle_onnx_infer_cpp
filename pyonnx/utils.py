import os
import glob
import json
import time

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
  def __init__(self, gt_json_path,model_type):
    self.dt_json_path = gt_json_path.replace(".json",f"_dt_{model_type}.json")
    self.perf_json_path = gt_json_path.replace(".json",f"_perf_{model_type}.json")
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
    save_json(self.perf_info,   self.perf_json_path)

  

if __name__ == "__main__":
  read_images_from_gt("dataset/annotations/instances_default.json")