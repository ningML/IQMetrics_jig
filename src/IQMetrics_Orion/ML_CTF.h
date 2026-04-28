#pragma once
#ifdef IQMETRICS_EXPORTS
#define IQMETRICS_API __declspec(dllexport)
#else
#define IQMETRICS_API __declspec(dllimport)
#endif
//#include "ml_image_public.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include<iostream>
//#include "IQMetricUtl.h"
namespace MLIQMetrics
{
    enum CTFType
    {
        CTF_VERTICAL,
        CTF_HORIZONTAL
    };
struct CTFRe
{
    double dof = 0;
    double maxCentralContrast = 0;
    double uniformityAtCenteralFocus = 0;
    double deltaZ = 0;
    std::vector<double>ctfTL;
    std::vector<double>ctfTR;
    std::vector<double>ctfBL;
    std::vector<double>ctfBR;
    std::vector<double>ctfCen;
    std::vector<double>zVec;
    std::string orientation = "";
    cv::Mat ctfMat;
    cv::Mat imgdraw;
    bool flag = true;
    std::string errMsg = "";
};
class IQMETRICS_API MLCTF 
{
  public:
    MLCTF();
    ~MLCTF();
  public:
      void setGap(int gap);
      void setROILen(int len);
    CTFRe getCTF(cv::Mat img,std::string savePath);
    CTFRe getCTF(std::string path, std::string savePath);
    double calculateCTF(cv::Mat roi, CTFType type);
  private:
      std::string getCTFOrientation(cv::Mat roi, CTFType&type);
      void writeResultToCSV(std::string saveFile, CTFRe re);
      void writeCTFResultToCSV(std::string saveFile, CTFRe re);
      bool pythonPlot(std::string path);
      double calculateFocuceUniformity(CTFRe re);
      double calculateDOF(CTFRe re,std::vector<double>zVec,double&deltZ);
    int m_ROILen = 105;
    int m_gap = 5;
};
} // namespace MLIQMetrics
