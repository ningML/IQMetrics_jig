#pragma once
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include<iostream>

#ifdef IQMETRICS_EXPORTS
#define IQMETRICS_API __declspec(dllexport)
#else
#define IQMETRICS_API __declspec(dllimport)
#endif
namespace MLIQMetrics {
	class IQMETRICS_API IQMetricsParameters
	{
	public:
		static double pixel_size;
		static double FocalLength;
	};

	class IQMETRICS_API IQMetricUtl
	{
	public:
		IQMetricUtl();
		~IQMetricUtl();
		static IQMetricUtl* instance();
	public:
		bool readImgsInfo(std::string path, std::vector<cv::Mat>& imgVec, std::vector<double>& zVec);
		int getBinNum(cv::Size s);
		double getPix2Arcmin(cv::Size s);
		double getPix2Degree(cv::Size s);
		static bool isInitFromJson;
		void loadJsonConfig(const char* path);
		void parseFilename(const std::string& name, int& first, double& last);

	private:
		static IQMetricUtl* self;

	};
}

