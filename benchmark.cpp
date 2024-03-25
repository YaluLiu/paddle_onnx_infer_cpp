#include "benchmark.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include <iostream>
#include <regex>

std::string readJsonfile(std::string path)
{
    using namespace rapidjson;
    std::ifstream config_file(path);
    if (!config_file.is_open())
    {
        return "json file not exist";
    }
 
    IStreamWrapper config(config_file);
    Document doc;
    doc.ParseStream(config);
 
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);
 
    config_file.close();
    return buffer.GetString();
}
 
void writeToJsonFile(std::string &jsonstr, std::string filepath)
{
    using namespace rapidjson;
    Document doc;
    doc.Parse(jsonstr.c_str());
 
    FILE* fp = fopen(filepath.c_str(), "wb");
    char writeBuffer[65535];
    FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
    PrettyWriter<FileWriteStream> writer(os);
    doc.Accept(writer);
    fclose(fp);
    std::cout << "writeToJsonFile end" <<std::endl;
}

void write_json(){
  rapidjson::StringBuffer strBuf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
  using namespace std;
  writer.StartObject();
  //1. 整数类型
  // writer.Key("Int");
  // writer.Int(1);

  // //2. 浮点类型
  // writer.Key("Double");
  // writer.Double(12.0000001);

  // //3. 字符串类型
  writer.Key("images");
  writer.String("xx.jpg");

  // //4. 结构体类型
  // writer.Key("Object");
  // writer.StartObject();
  // writer.Key("name");
  // writer.String("qq849635649");
  // writer.Key("age");
  // writer.Int(25);
  // writer.EndObject();

  //5.2 浮点型数组
  writer.Key("DoubleArray");
  writer.StartArray();
  for(int i = 1; i < 4; i++)
  {
      writer.Double(i * 1.0);
  }
  writer.EndArray();

  // //5.3 字符串数组
  // writer.Key("StringArray");
  // writer.StartArray();
  // writer.String("one");
  // writer.String("two");
  // writer.String("three");
  // writer.EndArray();

  //5.4 混合型数组
  //这说明了，一个json数组内容是不限制类型的
  // writer.Key("MixedArray");
  // writer.StartArray();
  // writer.String("one");
  // writer.Int(50);
  // writer.Bool(false);
  // writer.Double(12.005);
  // writer.EndArray();

  //5.5 结构体数组
  writer.Key("People");
  writer.StartArray();
  for(int i = 0; i < 3; i++)
  {
      writer.StartObject();
      writer.Key("name");
      writer.String("qq849635649");
      writer.Key("age");
      writer.Int(i * 10);
      writer.Key("sex");
      writer.Bool((i % 2) == 0);
      writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();

  string data = strBuf.GetString();
  writeToJsonFile(data,"test.json");
}

void read_json(std::vector<std::string>& image_name_list,std::vector<int>& image_id_list){
  std::string json_path = "/ultralytics/dataset/annotations/instances_default.json";
  std::string data = readJsonfile(json_path);
  // using namespace rapidjson;
  rapidjson::Document doc;
  doc.Parse(data.data());

  if(doc.HasParseError()) {
    std::cout << "Error: gt json parse error" <<  std::endl;
    return;
  }
  if(!doc.HasMember("images")) {
    std::cout << "Error: gt json not has images array" << std::endl;
    return;
  }
  if(!doc["images"].IsArray()){
    std::cout << "Error: gt json is not  array" << std::endl;
    return;
  }
  const rapidjson::Value& array = doc["images"];
  size_t len = array.Size();

  for(size_t i = 0; i < len; i++)
  {
      const rapidjson::Value& object = array[i];
      if(!object.IsObject()){
          continue;
      }
      int image_id = object["id"].GetInt();
      std::string image_name = object["file_name"].GetString();
      image_name_list.push_back(image_name);
      image_id_list.push_back(image_id);
  }
}

void BenchMark::init_model(){
}


void BenchMark::run(){
  
}