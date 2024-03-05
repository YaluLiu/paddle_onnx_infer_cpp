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

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


enum MODEL_TYPE
{
    //ultralytics
    //https://github.com/ultralytics/ultralytics
    YOLO_DETECT_V8 = 1,

    //paddledetection-baidu百度飞桨
    //https://github.com/PaddlePaddle/PaddleDetection
    YOLO_PADDLE=2,

    //FLOAT16 MODEL
    //no use for now
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
};


typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int	keyPointsNum = 2;  //Note:kpt number for pose
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;


typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;


class YOLO_V8
{
public:
    YOLO_V8();
    ~YOLO_V8();

public:
    char* CreateSession(DL_INIT_PARAM& iParams);
    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
    char* WarmUpSession();

    // infer function
    template<typename N>
    char* PaddleProcess(cv::Mat& iImg, N& blob, std::vector<DL_RESULT>& oResult);
    template<typename N>
    char* YOLO_origin_Process(cv::Mat& iImg, N& blob, std::vector<DL_RESULT>& oResult);

    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes{};

private:
    //must be class variable,else crush ??? 
    Ort::Env m_env;
    Ort::Session* m_session;
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;

    DL_INIT_PARAM m_params;
    float m_resizeScales;//letterbox scale

    float m_img_w,m_img_h; // input image size
};
