
#! /bin/bash

project_dir=$(cd "$(dirname "$0")";pwd)
container_name="ultralytics"
image_name="353942829/ultralytics:cpp"
work_dir="/ultralytics"

function download_onnx_runtime(){
  onnx_version="1.16.3"
  os="linux" # win
  wget https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-${os}-x64-gpu-${onnx_version}.tgz
  tar -zxvf onnxruntime-${os}-x64-gpu-${onnx_version}.tgz
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

#-------------------------------------------------------
# docker container command
#-------------------------------------------------------
container_name="cpp_onnx"
function restart(){
  docker restart ${container_name}
}

function exec() {
  docker exec -it ${container_name} bash
}

function stop() {
  docker stop ${container_name}
  docker rm ${container_name}
}

local_data_path=""
data_path=""
# -p 8888:8888 \
function create() {
    stop
    # # Run the ultralytics image in a container with GPU support
    docker run --user root --name ${container_name} --gpus all --shm-size=128g -it --privileged \
       -v ${project_dir}:${work_dir} \
       ${image_name}
    restart
    exec
}

function main() {
    if [ $# != 1 ] ; then
        echo "param:build or clean"
    else
        $1
    fi
}

main "$@"


