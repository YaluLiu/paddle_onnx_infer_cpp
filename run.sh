
#! /bin/bash

project_dir=$(cd "$(dirname "$0")";pwd)

function download_onnx_runtime(){
  onnx_version="1.16.3"
  wget https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-linux-x64-gpu-${onnx_version}.tgz
  tar -zxvf onnxruntime-linux-x64-gpu-${onnx_version}.tgz
  # for save visualize result
  mkdir result
}

function build(){
  mkdir build
  cmake -B build .
  run
}

function run(){
  cd build
  make 
  ./Yolov8OnnxRuntimeCPPInference
  cd ..
}

function main() {
    if [ $# != 1 ] ; then
        echo "param:build or clean"
    else
        $1
    fi
}

main "$@"


