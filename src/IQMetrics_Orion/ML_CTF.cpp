#include "pch.h"
#include "ML_CTF.h"
#include"LogPlus.h"
#include <filesystem>
#include <string>
#include <regex>
#include <algorithm>
#include <chrono>
#include <armadillo>
#include <embed.h>
#include <opencv2/core.hpp>
#include "ml_image_public.h"
#include "IQMetricUtl.h"


namespace py = pybind11;

namespace fs = std::filesystem;
using namespace MLImageDetection;
using namespace std;
using namespace MLIQMetrics;
using namespace cv;
struct FileInfo
{
	fs::path path;
	std::time_t time;
	int firstNum;
	double lastNum;
};
MLIQMetrics::MLCTF::MLCTF()
{
}

MLIQMetrics::MLCTF::~MLCTF()
{
}

void MLIQMetrics::MLCTF::writeResultToCSV(std::string saveFile, CTFRe re)
{
	std::ofstream file(saveFile);
	// 写表头
	file << "dof,maxCentralContrast,uniformityAtCentralFocus,deltaZ\n";
	// 写数据
	file << re.dof << "," << re.maxCentralContrast << "," << re.uniformityAtCenteralFocus << "," << re.deltaZ << "\n";
	file.close();


}

void MLIQMetrics::MLCTF::writeCTFResultToCSV(std::string saveFile, CTFRe re)
{
	std::ofstream file(saveFile);
	// 写表头
	file << "motorZloc(mm),ctfTL,ctfTR,ctfBL,ctfBR,ctfCen\n";
	// 写数据
	for (int i = 0; i < re.zVec.size(); i++)
	{
		file << re.zVec[i] << "," << re.ctfTL[i] << "," << re.ctfTR[i] << "," << re.ctfBL[i]<<","<<re.ctfBR[i]<<","<<re.ctfCen[i] << "\n";
	}
	file.close();
}

bool MLIQMetrics::MLCTF::pythonPlot(std::string path)
{
	py::scoped_interpreter guard{};  // 启动 Python 解释器

	try {
		py::module_ sys = py::module_::import("sys");
		string pythonPath = "E:\\project\\jig\\src\\app\\config\\pythonPlot";
		sys.attr("path").attr("insert")(0, pythonPath);
		py::module_ test = py::module_::import("ctfplot");
		//int result = test.attr("add")(2, 3).cast<int>();
		//std::cout << "add result = " << result << std::endl;
		std::string msg = test.attr("saveCTFFigure")(path).cast<std::string>();
		//std::string msg = test.attr("greet")("Tom").cast<std::string>();
	   //std::cout << msg << std::endl;
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

double MLIQMetrics::MLCTF::calculateFocuceUniformity(CTFRe re)
{
	vector<double>maxctfVec;
	maxctfVec.push_back(*max_element(re.ctfBL.begin(),re.ctfBL.end()));
	maxctfVec.push_back(*max_element(re.ctfBR.begin(), re.ctfBR.end()));
	maxctfVec.push_back(*max_element(re.ctfTL.begin(), re.ctfTL.end()));
	maxctfVec.push_back(*max_element(re.ctfTR.begin(), re.ctfTR.end()));
	maxctfVec.push_back(*max_element(re.ctfCen.begin(), re.ctfCen.end()));
	double minctf = *min_element(maxctfVec.begin(), maxctfVec.end());
	double maxctf = *max_element(maxctfVec.begin(), maxctfVec.end());
	return minctf/ maxctf;
}

double MLIQMetrics::MLCTF::calculateDOF(CTFRe re, vector<double> zVec, double& deltZ)
{
	vector<double>ctfCen = re.ctfCen;
	arma::vec ctfvec(ctfCen.data(), ctfCen.size(), false);
	arma::vec zvec(zVec.data(), zVec.size(), false);
	zvec = (zvec - zvec[0]) * 1000;
	vector<double>z1= arma::conv_to<std::vector<double>>::from(zvec);
	double maxctf = arma::max(ctfvec);
	arma::vec sub = ctfvec - maxctf * 0.85;
	sub = arma::abs(sub);
	arma::uvec idx = arma::sort_index(sub);
	double loc1 = zvec[idx[0]];
	double loc2 = zvec[idx[1]];
	double dof = max(loc1,loc2)-min(loc1,loc2);
	double cen = (loc1 + loc2) / 2.0;
	//arma::uword idx = ctfvec.index_max();
	arma::uvec idx1 = arma::find(ctfvec == maxctf);
	deltZ = cen - zvec(idx1[0]);
	return dof;
}

std::string MLIQMetrics::MLCTF::getCTFOrientation(cv::Mat roi, CTFType& type)
{
	cv::Mat rowmat, colmat;
	cv::reduce(roi, rowmat, 0, REDUCE_AVG);
	cv::reduce(roi, colmat, 1, REDUCE_AVG);
	string orien = "";
	double maxRow, minRow, maxCol, minCol;
	cv::minMaxLoc(rowmat, &minRow, &maxRow);
	cv::minMaxLoc(colmat, &minCol, &maxCol);
	double ratioR = maxRow / minRow;
	double ratioC = maxCol / minCol;
	if (ratioC > ratioR)
	{
		orien = "Horizontal";
		type = CTF_HORIZONTAL;
	}
	else
	{
		orien = "Vertical";
		type = CTF_VERTICAL;
	}

	return orien;
}


double MLIQMetrics::MLCTF::calculateCTF(cv::Mat roi, CTFType type)
{
	cv::Mat rowMat;
	if (type == CTF_VERTICAL)
		cv::reduce(roi, rowMat, 0, REDUCE_AVG);
	else
		cv::reduce(roi, rowMat, 1, REDUCE_AVG);
	double minVal, maxVal;
	cv::minMaxLoc(rowMat, &minVal, &maxVal);
	double ctf0 = (maxVal - minVal) / (maxVal + minVal + 1e-6);
	//double ctf0 = maxVal / (maxVal+ minVal);
	return ctf0;
}

void MLIQMetrics::MLCTF::setGap(int gap)
{
	m_gap = gap;
}

void MLIQMetrics::MLCTF::setROILen(int len)
{
	m_ROILen = len;
}

CTFRe MLIQMetrics::MLCTF::getCTF(cv::Mat img, std::string savePath)
{
	string info = "---getCTF---";
	string message = info + "Begining grid contrast test ";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	CTFRe re;
	if (img.empty())
	{
		re.flag = false;
		re.errMsg = info + "Input image is NULL";
		LOG4CPLUS_ERROR(LogPlus::getInstance()->logger, re.errMsg.c_str());
		return re;
	}
	MLimagePublic pl;
	cv::Mat img8 = pl.convertToUint8(img);
	cv::Mat imgdraw = pl.convertTo3Channels(img8);
	int row = img.rows;
	int col = img.cols;
	cv::Rect rectTL(m_gap, m_gap, m_ROILen, m_ROILen);
	cv::Rect rectTR(col - m_gap - m_ROILen, m_gap, m_ROILen, m_ROILen);
	cv::Rect rectBL(m_gap, row - m_gap - m_ROILen, m_ROILen, m_ROILen);
	cv::Rect rectBR(col - m_gap - m_ROILen, row - m_gap - m_ROILen, m_ROILen, m_ROILen);
	cv::Rect rectCen(col / 2 - m_ROILen / 2, row / 2 - m_ROILen / 2, m_ROILen, m_ROILen);
	cv::rectangle(imgdraw, rectTL, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectTR, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectBL, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectBR, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectCen, Scalar(0, 0, 255), 5);
	//TODO: has problems
	//  Determining grid spatial frequency 
	// Grid spatial frequency: 6   
	message = info + "Determining grid orientation";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	CTFType type;
	string orientation = getCTFOrientation(img(rectCen).clone(), type);
	message = info + " Grid orientation:" + orientation;
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	message = info + "Processing focus curves";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	double ctfTL = calculateCTF(img(rectTL).clone(), type);
	double ctfTR = calculateCTF(img(rectTR).clone(), type);
	double ctfBL = calculateCTF(img(rectBL).clone(), type);
	double ctfBR = calculateCTF(img(rectBR).clone(), type);
	double ctfCen = calculateCTF(img(rectCen).clone(), type);
	re.ctfTL.push_back(ctfTL);
	re.ctfTR.push_back(ctfTR);
	re.ctfBL.push_back(ctfBL);
	re.ctfBR.push_back(ctfBR);
	re.ctfCen.push_back(ctfCen);
	re.maxCentralContrast = ctfCen;
	re.orientation = orientation;
	re.imgdraw = imgdraw;	
	//TODO: save plot
	fs::path fullPath = fs::path(savePath) / orientation;
	if (!fs::exists(fullPath))
	{
		fs::create_directories(fullPath);
	}
	std::string file = (fullPath / "ROIs.tif").string();
	cv::imwrite(file, re.imgdraw);
	std::string csvfile = (fullPath / "Results.csv").string();
	writeResultToCSV(csvfile, re);
	std::string ctffile = (fullPath / "ctfdetails.csv").string();

	message = info + "Saving result images ";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	message = info + "Test complete";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	return re;
}

CTFRe MLIQMetrics::MLCTF::getCTF(std::string path, std::string savePath)
{
	string info = "---getCTF---";
	string message = info + "Begining grid contrast test ";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	CTFRe re;
	if (!fs::exists(path))
	{
		re.flag = false;
		re.errMsg = info + "Input filePath does not exist!!!";
		LOG4CPLUS_ERROR(LogPlus::getInstance()->logger, re.errMsg.c_str());
		return re;
	}
	vector<double>zVec;
	vector<cv::Mat>imgVec;
	IQMetricUtl utl;
	utl.readImgsInfo(path,imgVec,zVec);
	if (imgVec.size() < 1)
	{
		re.flag = false;
		re.errMsg = info + "There is no imgs in the filePath!!!";
		LOG4CPLUS_ERROR(LogPlus::getInstance()->logger, re.errMsg.c_str());
		return re;
	}
	vector<double>ctfTLVec, ctfTRVec, ctfBLVec, ctfBRVec, ctfCenVec;
	cv::Mat imgdraw;
	string orientation;
	MLimagePublic pl;
	cv::Mat img = imgVec[0];
	cv::Mat img8 = pl.convertToUint8(img);
	imgdraw = pl.convertTo3Channels(img8);
	int row = img.rows;
	int col = img.cols;
	cv::Rect rectTL(m_gap, m_gap, m_ROILen, m_ROILen);
	cv::Rect rectTR(col - m_gap - m_ROILen, m_gap, m_ROILen, m_ROILen);
	cv::Rect rectBL(m_gap, row - m_gap - m_ROILen, m_ROILen, m_ROILen);
	cv::Rect rectBR(col - m_gap - m_ROILen, row - m_gap - m_ROILen, m_ROILen, m_ROILen);
	cv::Rect rectCen(col / 2 - m_ROILen / 2, row / 2 - m_ROILen / 2, m_ROILen, m_ROILen);
	cv::rectangle(imgdraw, rectTL, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectTR, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectBL, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectBR, Scalar(0, 0, 255), 5);
	cv::rectangle(imgdraw, rectCen, Scalar(0, 0, 255), 5);
	for (int i = 0; i < imgVec.size(); i++)
	{
		cv::Mat img = imgVec[i];
		//cv::Mat img8 = pl.convertToUint8(img);
		//imgdraw = pl.convertTo3Channels(img8);
		//int row = img.rows;
		//int col = img.cols;
		//cv::Rect rectTL(m_gap, m_gap, m_ROILen, m_ROILen);
		//cv::Rect rectTR(col - m_gap - m_ROILen, m_gap, m_ROILen, m_ROILen);
		//cv::Rect rectBL(m_gap, row - m_gap - m_ROILen, m_ROILen, m_ROILen);
		//cv::Rect rectBR(col - m_gap - m_ROILen, row - m_gap - m_ROILen, m_ROILen, m_ROILen);
		//cv::Rect rectCen(col / 2 - m_ROILen / 2, row / 2 - m_ROILen / 2, m_ROILen, m_ROILen);
		//cv::rectangle(imgdraw, rectTL, Scalar(0, 0, 255), 5);
		//cv::rectangle(imgdraw, rectTR, Scalar(0, 0, 255), 5);
		//cv::rectangle(imgdraw, rectBL, Scalar(0, 0, 255), 5);
		//cv::rectangle(imgdraw, rectBR, Scalar(0, 0, 255), 5);
		//cv::rectangle(imgdraw, rectCen, Scalar(0, 0, 255), 5);
		//TODO: has problems
		//  Determining grid spatial frequency 
		// Grid spatial frequency: 6   
		message = info + "Determining grid orientation";
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
		CTFType type;
		orientation = getCTFOrientation(img(rectCen).clone(), type);
		message = info + " Grid orientation:" + orientation;
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
		message = info + "Processing focus curves";
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
		double ctfTL = calculateCTF(img(rectTL).clone(), type);
		double ctfTR = calculateCTF(img(rectTR).clone(), type);
		double ctfBL = calculateCTF(img(rectBL).clone(), type);
		double ctfBR = calculateCTF(img(rectBR).clone(), type);
		double ctfCen = calculateCTF(img(rectCen).clone(), type);
		ctfTLVec.push_back(ctfTL);
		ctfTRVec.push_back(ctfTR);
		ctfBLVec.push_back(ctfBL);
		ctfBRVec.push_back(ctfBR);
		ctfCenVec.push_back(ctfCen);
	}
	re.ctfTL=ctfTLVec;
	re.ctfTR=ctfTRVec;
	re.ctfBL=ctfBLVec;
	re.ctfBR=ctfBRVec;
	re.ctfCen=ctfCenVec;
	re.orientation = orientation;
	re.maxCentralContrast = *max_element(ctfCenVec.begin(),ctfCenVec.end());
	re.uniformityAtCenteralFocus = calculateFocuceUniformity(re);
	re.dof = calculateDOF(re, zVec, re.deltaZ);
	re.imgdraw = imgdraw;
	re.zVec = zVec;
	//TODO: save plot
	fs::path fullPath = fs::path(savePath) / orientation;
	if (!fs::exists(fullPath))
	{
		fs::create_directories(fullPath);
	}
	std::string csvfile = (fullPath / "Results.csv").string();
	writeResultToCSV(csvfile, re);
	std::string ctffile = (fullPath / "CTFResults.csv").string();
	writeCTFResultToCSV(ctffile, re);
	message = info + "Exporting results.";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	std::string file = (fullPath / "ROIs.tif").string();
	cv::imwrite(file, re.imgdraw);
	bool flag=pythonPlot(fullPath.string());
	if (flag)
	{
		message = info + "Saving result images ";
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	}
	else
	{
		message = info + "Saving result images fail";
		LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	}
	message = info + "Test complete";
	LOG4CPLUS_INFO(LogPlus::getInstance()->logger, message.c_str());
	return re;
}



