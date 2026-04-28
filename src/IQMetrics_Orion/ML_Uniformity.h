#pragma once
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include<iostream>
#ifdef IQMETRICS_EXPORTS
#define IQMETRICS_API __declspec(dllexport)
#else
#define IQMETRICS_API __declspec(dllimport)
#endif
namespace MLIQMetrics
{
	struct UniformityRe
	{
		double uniformity = 0;
		double symmetry = 0;
		cv::Mat imgResult;
		cv::Mat imgEnhanced;
		cv::Mat imgdraw;
		bool flag = true;
		std::string errMsg = "";

	};
	class IQMETRICS_API MLUniformity
	{
	public:
		MLUniformity();
		~MLUniformity();
	public:
		void setUniformityROI(cv::Rect rect);
		void setSmoothLen(int len);
		void setDarkLevel(int dark);
		UniformityRe getUniformity(std::string filePath, std::string savePath);
	private:
		double calculateSymmetry(cv::Mat img, cv::Mat imgR);
		void writeResultToCSV(std::string saveFile, double uniformity, double symmetry);
		bool pythonPlot(std::string path);
		cv::Rect m_Rect = cv::Rect(0, 0, -1, -1);
		int m_smooth = 11;
		int m_dark = 0;
	};
}

