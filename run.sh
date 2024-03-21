
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

# bash eval paddle or yolo
function eval(){
  dataset_dir="dataset"
  anno_dir=${dataset_dir}/annotations
  python pyonnx/eval.py --gt ${anno_dir}/instances_default.json --dt ${anno_dir}/instances_default_dt_yolo.json
  python pyonnx/eval.py --gt ${anno_dir}/instances_default.json --dt ${anno_dir}/instances_default_dt_paddle.json
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


local_dir="dataset"
function mount() {
  server_name="yalu"
  server_pwd="liuyalu4545"
  local_name="boli-shixi"
  server_dir="//192.168.203.3/BoLiTech/RDTeam/01_BDAI/标准测试集/标准数据集01"

  sudo mount -t cifs -o rw,uid=${local_name},user=${server_name},pass=${server_pwd} ${server_dir} ${local_dir}
}

function umount() {
  umount ${local_dir}
}

# -p 8888:8888 \
function create() {
    stop
    # # Run the ultralytics image in a container with GPU support
    docker run --user root --name ${container_name} --gpus all --shm-size=128g -it --privileged \
       -v ${project_dir}:${work_dir} \
       -v /home/boli-shixi/yalu/ultralytics/models:${work_dir}/models \
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


