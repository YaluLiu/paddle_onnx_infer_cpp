#include <iostream>
#include <iomanip>
#include "yolo_net.h"
#include <random>
#include <string>



void DetectTest()
{
  std::string img_path = std::string("images/bus.jpg");
  cv::Mat img = cv::imread(img_path);
  YOLO_NET net;
  std::vector<DL_RESULT> res;
  net.predict(img,res);
  net.visualize(img,res);
}



int main(int argc, char *argv[])
{
  DetectTest();
}
