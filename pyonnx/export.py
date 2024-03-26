from ultralytics import YOLO

def export_onnx(model):
  # Use the model
  model.export(
      format='onnx', 
      imgsz=(640,640), 
      half=False, 
      dynamic=True, 
      simplify=False, 
      opset=12, 
      device=0,
      verbose=False)

def export_engine(model):
  model.export(
      format='engine', 
      imgsz=(640,640), 
      half=False, 
      dynamic=True, 
      simplify=False, 
      opset=12, 
      device=0,
      verbose=False)

if __name__ == "__main__":
  model = YOLO("models/yolov8n.pt")
  export_engine(model)
  export_onnx(model)