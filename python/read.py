import pickle
import json
import numpy as np
import cv2
import pdb


json_name="instances_val2014.json"
with open(json_name, 'r') as f:
  val_json = json.loads(f.read())
  print(val_json.keys()) # ['info', 'images', 'licenses', 'annotations', 'categories']

with open("x.json","r") as f:
  x = json.loads(f.read())

val_json['annotations'] = val_json['annotations'][:10]
val_json['images'] = val_json['images'][:10]

for i in range(10):
  val_json['annotations'][i]["image_id"] = i
  val_json['images'][i] = {"id":i}

val_json["info"] = []
val_json["licenses"] = []
val_json["categories"] = []

with open("x_val.json", 'w') as f:
  json.dump(val_json,f,indent=2)