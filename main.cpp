#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include "benchmark.h"

int trans_class_id(int class_id){
  if (class_id == 0){
    return 1;
  }
  
  if (class_id == 2 || class_id == 5 || class_id == 7){
    return 2;
  }
    
  
  if (class_id == 1) {
    return 3;
  }
  
  if (class_id == 3) {
    return 4;
  }
  
  if (class_id == 15 or class_id == 16) {
    return 5;
  }
  return -1;
}

void Detector(YOLO_V8*& p,std::vector<std::string> image_name_list,std::vector<int> image_id_list,std::string model_type) {
    size_t num_images = image_name_list.size();
    rapidjson::StringBuffer anno_str_buf;
    rapidjson::Writer<rapidjson::StringBuffer> anno_writer(anno_str_buf);

    anno_writer.StartArray();
    for (size_t i = 0; i < num_images; ++i)
    {
        std::string image_name = image_name_list[i];
        std::string img_path = "../dataset/images/" + image_name;
        cv::Mat img = cv::imread(img_path);
        std::vector<DL_RESULT> res;
        p->RunSession(img, res);
        for (auto& re : res)
        {
            cv::RNG rng(cv::getTickCount());
            cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

            cv::rectangle(img, re.box, color, 3);

            float confidence = floor(100 * re.confidence) / 100;
            std::cout << std::fixed << std::setprecision(2);
            std::string label = p->classes[re.classId] + " " +
                std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

            cv::rectangle(
                img,
                cv::Point(re.box.x, re.box.y - 25),
                cv::Point(re.box.x + label.length() * 15, re.box.y),
                color,
                cv::FILLED
            );

            cv::putText(
                img,
                label,
                cv::Point(re.box.x, re.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 0, 0),
                2
            );

            //{"image_id": 1, "bbox": [836.0142822265625, 324.4328308105469, 296.3536376953125, 114.25949096679688], "category_id": 2, "score": 0.8}
            int image_id = image_id_list[i];
            int x = re.box.x;
            int y = re.box.y;
            int width = re.box.width;
            int height = re.box.height;
            int class_id = re.classId;
            float conf = re.confidence;
            class_id = trans_class_id(class_id);
            if (class_id == -1){
              continue;
            }
            anno_writer.StartObject();
            anno_writer.Key("image_id");
            anno_writer.Int(image_id);
            anno_writer.Key("category_id");
            anno_writer.Int(class_id);
            anno_writer.Key("score");
            anno_writer.Double(conf);

            anno_writer.Key("bbox");
            anno_writer.StartArray();
            anno_writer.Int(x);anno_writer.Int(y);anno_writer.Int(width);anno_writer.Int(height);
            anno_writer.EndArray();
            anno_writer.EndObject();
        }
        std::string result_path=std::string("../dataset/result/") + image_name;
        cv::imwrite(result_path,img);
    }
    anno_writer.EndArray();
    std::string anno_json_path = "../dataset/annotations/dt_" + model_type + "_cpp.json";
    std::string data = anno_str_buf.GetString();
    writeToJsonFile(data,anno_json_path);

    rapidjson::StringBuffer perf_str_buf;
    rapidjson::Writer<rapidjson::StringBuffer> perf_writer(perf_str_buf);
    perf_writer.StartObject();
    perf_writer.Key("prev");
    perf_writer.Double(p->m_prev_cost);
    perf_writer.Key("infer");
    perf_writer.Double(p->m_infer_cost);
    perf_writer.Key("post");
    perf_writer.Double(p->m_post_cost);
    perf_writer.EndObject();
    std::string perf_json_path = "../dataset/annotations/perf_" + model_type + "_cpp.json";
    data = perf_str_buf.GetString();
    writeToJsonFile(data,perf_json_path);
  }



int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}

void DetectTest(std::string model_type, std::string model_name)
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.6;
    params.iouThreshold = 0.5;
    params.modelPath = std::string("/ultralytics/models/")+ model_name;
    if (model_type == "yolo"){
      params.modelType = YOLO_DETECT_V8;
    } else if (model_type == "paddle"){
      params.modelType = YOLO_PADDLE;
    } else{
      std::cout << "error on model_type" << std::endl;
      return;
    }
    

    // params.modelPath = "/home/boli-shixi/yalu/ultralytics/models/yolov8_n_500e_coco.onnx";
    // params.modelPath = "/home/boli-shixi/yalu/ultralytics/models/ppyoloe_plus_crn_l_80e_coco.onnx";

    params.imgSize = { 640, 640 };
    
#ifdef USE_CUDA
    std::cout << "use_cuda" << params.modelPath << std::endl;
    params.cudaEnable = true;
#else
    std::cout << "use_cpu" << params.modelPath << std::endl;
    params.cudaEnable = false;

#endif
    yoloDetector->CreateSession(params);
    std::vector<std::string> image_name_list;
    std::vector<int> image_id_list;
    read_json(image_name_list,image_id_list);
    // for(size_t i = 0; i < image_name_list.size();++i){
    //   std::cout << i << ":" << image_id_list[i] << "," << image_name_list[i] << std::endl;
    // }
    Detector(yoloDetector,image_name_list,image_id_list,model_type);
}



int main(int argc, char *argv[])
{
  std::string model_type;
  std::string model_name;
  if (argc == 3){
    model_type = argv[1];
    model_name = argv[2];
  } else if(argc < 3){
    std::cout << "Error,please input model_type and model_name,as ./YoloOnnx yolo yolov8n.onnx";
  }
  
  DetectTest(model_type,model_name);
}
