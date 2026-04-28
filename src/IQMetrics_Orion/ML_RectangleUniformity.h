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
	class IQMETRICS_API MLRectangleUniformity
	{
	public:
		MLRectangleUniformity();
		~MLRectangleUniformity();
	};
}
