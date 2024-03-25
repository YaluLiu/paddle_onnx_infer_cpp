#pragma once

#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "inference.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

#include <filesystem>
#include <fstream>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/encodedstream.h"
#include <cstdio>

std::string readJsonfile(std::string path);
void write_json();

void read_json(std::vector<std::string>& image_name_list,std::vector<int>& image_id_list);
void writeToJsonFile(std::string &jsonstr, std::string filepath);

class BenchMark
{
public:
    BenchMark();
    ~BenchMark();
public:
    void init_model();
    void run();
private:
    YOLO_V8* m_model;
};
