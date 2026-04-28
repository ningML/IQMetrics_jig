#include "pch.h"
#include "ML_Uniformity.h"
#include <filesystem>
#include"LogPlus.h"
#include "IQMetricUtl.h"
#include"ml_image_public.h"
#include <embed.h>
namespace py = pybind11;

using namespace cv;
using namespace std;
using namespace MLIQMetrics;
namespace fs = std::filesystem;

MLIQMetrics::MLUniformity::MLUniformity()
{
}

MLIQMetrics::MLUniformity::~MLUniformity()
{
}

void MLIQMetrics::MLUniformity::setUniformityROI(cv::Rect rect)
{
	m_Rect = rect;
}

void MLIQMetrics::MLUniformity::setSmoothLen(int len)
{
	m_smooth = len;
}

void MLIQMetrics::MLUniformity::setDarkLevel(int dark)
{
	m_dark = dark;
}

UniformityRe MLIQMetrics::MLUniformity::getUniformity(std::string filePath, std::string savePath)
{
	string info = "---getUniformity---";
	string message = info + "Begining Uniformity and symmetry test ";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	UniformityRe re;
	if (!fs::exists(filePath))
	{
		re.flag = false;
		re.errMsg = info + "Input filePath does not exist!!!";
		LOG4CPLUS_ERROR(LogPlus::getInstance()->logger, re.errMsg.c_str());
		return re;
	}
	vector<double>zVec;
	vector<cv::Mat>imgVec;
	IQMetricUtl utl;
	 message = info + " Loading images  start ";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	utl.readImgsInfo(filePath, imgVec, zVec);
	 message = info + " Loading images end: the num is "+to_string(imgVec.size());
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	if (imgVec.size() < 1)
	{
		re.flag = false;
		re.errMsg = info + "There is no imgs in the filePath!!!";
		LOG4CPLUS_ERROR(LogPlus::getInstance()->logger, re.errMsg.c_str());
		return re;
	}
	cv::Mat imgInfer(imgVec[0].size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < imgVec.size(); i++)
	{
		cv::Mat imgTmp = imgVec[i];
		imgTmp = imgTmp - m_dark;
		cv::medianBlur(imgTmp, imgTmp, m_smooth);
		//cv::add(imgTmp, imgInfer, imgInfer);
		imgTmp.convertTo(imgTmp, CV_32FC1);
		imgInfer = imgInfer + imgTmp;
	}
	message = info + " Producing median image.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	imgInfer = imgInfer / imgVec.size();
	imgInfer.convertTo(imgInfer, CV_8UC1);
	re.imgResult = imgInfer;
	cv::Mat imgEnhance;
	cv::equalizeHist(imgInfer, imgEnhance);
	re.imgEnhanced = imgEnhance;
	message = info + " Smoothing image.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());

	MLImageDetection::MLimagePublic pl;
	cv::Mat roi = pl.getRectROIImg(imgInfer, m_Rect);
	double minV, maxV;
	cv::minMaxLoc(roi, &minV, &maxV);
	double uniformity = minV / maxV;
	re.uniformity = uniformity;
	message = info + " Calculating uniformity.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());

	//cv::Rect rect(0, imgInfer.rows - 500, 500, 500);
	//cv::rectangle(imgInfer, rect, Scalar(255, 255, 255));
	cv::Mat imgRx, imgRy, imgR;
	cv::flip(imgInfer, imgRx, 1);
	cv::flip(imgInfer, imgRy, 0);
	cv::rotate(imgInfer, imgR, ROTATE_180);
	double s1 = calculateSymmetry(imgInfer, imgRx);
	double s2 = calculateSymmetry(imgInfer, imgRy);
	double s3 = calculateSymmetry(imgInfer, imgR);
	double s = (s1 + s2 + s3) / 3.0;
	re.symmetry = s;
	message = info + "Calculating symmetry.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());

	std::string saveFolder = "Results";
	fs::path fullPath = fs::path(savePath) / saveFolder;
	if (!fs::exists(fullPath))
	{
		fs::create_directories(fullPath);
	}

	std::string fileRe = (fullPath / "Result Image.tif").string();
	cv::imwrite(fileRe, re.imgResult);
	std::string fileEn = (fullPath / "Result Enhanced Image.tif").string();
	cv::imwrite(fileEn, re.imgEnhanced);

	bool plotflag=pythonPlot(fullPath.string());
	if (plotflag)
	{
		message = info + "Saving result image.";
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	}
	else
	{
		message = info + "Saving result image fail.";
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	}

	std::string csvfile = (fullPath / "Results.csv").string();
	writeResultToCSV(csvfile, uniformity,s);
	message = info + "Exporting results.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	message = info + "Test complete.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	return re;
}

double MLIQMetrics::MLUniformity::calculateSymmetry(cv::Mat img, cv::Mat imgR)
{
	img.convertTo(img, CV_32FC1);
	imgR.convertTo(imgR, CV_32FC1);
	double num = 0, d1 = 0, d2 = 0;
	for (int y = 0; y < img.rows; y++) {
		const float* p1 = img.ptr<float>(y);
		const float* p2 = imgR.ptr<float>(y);
		for (int x = 0; x < img.cols; x++)
		{
			float a = p1[x];
			float b = p2[x];

			num += a * b;
			d1 += a * a;
			d2 += b * b;
		}
	}
	return num / sqrt(d1 * d2);
}

void MLIQMetrics::MLUniformity::writeResultToCSV(string saveFile, double uniformity, double symmetry)
{
	std::ofstream file(saveFile);
	file << "uniformity,symmetry\n";
	file << uniformity << "," << symmetry << "\n";
	file.close();
}

bool MLIQMetrics::MLUniformity::pythonPlot(std::string path)
{
	py::scoped_interpreter guard{};  // Æô¶¯ Python ½âÊÍÆ÷
	try {
		py::module_ sys = py::module_::import("sys");
		string pythonPath = "E:\\project\\jig\\src\\app\\config\\pythonPlot";
		sys.attr("path").attr("insert")(0, pythonPath);
		py::module_ test = py::module_::import("IlluminationPlot");
		int result = test.attr("add")(2, 3).cast<int>();
		std::cout << "add result = " << result << std::endl;
		std::string msg = test.attr("plotIlluminationResult")(path).cast<std::string>();
	   std::cout << msg << std::endl;
		return true;
	}
	catch (const std::exception& e)
	{
		string errInfo = e.what();
		LOG4CPLUS_ERROR(LogPlus::getInstance()->logger, "Python plot error");
		std::cout << "Error: " << e.what() << std::endl;
		return false;
	}
}
