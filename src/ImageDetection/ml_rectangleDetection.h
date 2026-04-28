#pragma once
#include "ml_image_public.h"
//#include <MLImageDetection>
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
using namespace std;
using namespace cv;
namespace MLImageDetection
{
class ALGORITHM_API RectangleDetection:public MLimagePublic
{
  public:
    RectangleDetection();
    ~RectangleDetection();

  public:
    cv::RotatedRect getRectangleBorder(cv::Mat img);
    cv::Rect getSolidExactRect(cv::Mat img8, cv::Rect rect);
    cv::Mat getImgdraw();

  private:
    cv::Mat m_img_draw;
};
} // namespace MLImageDetection
