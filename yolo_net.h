#pragma once
#include "inference.h"

class YOLO_NET
{
public:
    YOLO_NET();
    ~YOLO_NET();

public:
  void predict(cv::Mat& img, std::vector<DL_RESULT>& res);
  void visualize(cv::Mat& img, std::vector<DL_RESULT>& res);
private:
  YOLO_V8* m_yoloDetector;
  void Init();
  int ReadCocoYaml();
};