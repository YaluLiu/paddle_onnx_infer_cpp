# YOLOv8-onnx Inference C++

This example demonstrates how to perform inference using YOLOv8-onnx-models,now support:
[paddledetection-v2.7](https://github.com/PaddlePaddle/PaddleDetection) 
[ultralytics-v8.1](https://github.com/ultralytics/ultralytics)



## Usage

```bash
# download onnx_runtime_lib
bash run.sh download_runtime_lib
# must change mount dir before 
bash run.sh mount
# create docker
bash run.sh create
bash run.sh build
# export model of engine & onnx, it depends on the hardware environment 
bash run.sh export
bash run.sh run
```

```bash
# 先去把paddledetection的tensorRT版本跑了
bash run.sh eval
```


## more version of onnx-runtime-lib
[download_url](https://github.com/microsoft/onnxruntime/releases)


## your own 
if you want to load your own onnx model, may need [onnx-runtime](https://github.com/microsoft/onnxruntime)

