# YOLOv8-onnx Inference C++

This example demonstrates how to perform inference using YOLOv8-onnx-models,now support:
[paddledetection-v2.7](https://github.com/PaddlePaddle/PaddleDetection) 
[ultralytics-v8.1](https://github.com/ultralytics/ultralytics)


## Usage

```bash
# download onnx_runtime_lib
bash run.sh download_onnx_runtime
bash run.sh build
bash run.sh run
```

## your own 
if you want to load your own onnx model, may need [onnx-runtime](https://github.com/microsoft/onnxruntime)

