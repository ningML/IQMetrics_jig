#include "pch.h"
#include "IQMetricUtl.h"
#include <json.hpp>
#include<fstream>
#include"ml_image_public.h"
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;
using namespace std;
using Json = nlohmann::json;
using namespace MLImageDetection;
using namespace MLIQMetrics;
double IQMetricsParameters::pixel_size = 3.2e-3;
double IQMetricsParameters::FocalLength = 40;
bool IQMetricUtl::isInitFromJson = false;
IQMetricUtl* IQMetricUtl::self = nullptr;
struct FileInfo
{
	fs::path path;
	std::time_t time;
	int firstNum;
	double lastNum;
};

IQMetricUtl::IQMetricUtl()
{
	if (!IQMetricUtl::isInitFromJson)
	{
		MLimagePublic pl;

		string filepath = "./config/IQMetricsParametersConfig.json";
		loadJsonConfig(filepath.c_str());
		IQMetricUtl::isInitFromJson = true;
	}
}
IQMetricUtl::~IQMetricUtl()
{
}
IQMetricUtl* IQMetricUtl::instance()
{
	if (self == nullptr) {
		self = new IQMetricUtl();
	}
	return self;
	//return nullptr;
}

bool MLIQMetrics::IQMetricUtl::readImgsInfo(std::string folder, std::vector<cv::Mat>& imgVec, std::vector<double>& zVec)
{
	std::vector<FileInfo> files;
	for (auto& entry : fs::directory_iterator(folder))
	{
		if (entry.is_regular_file())
		{
			auto path = entry.path();

			if (path.extension() == ".tif")
			{
				FileInfo info;
				info.path = path;

				// 时间（最后修改时间）
				//info.time = decltype(entry.last_write_time())::clock::to_time_t(entry.last_write_time());
				auto ftime = entry.last_write_time();
				auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
					ftime - fs::file_time_type::clock::now()
					+ std::chrono::system_clock::now());
				info.time = std::chrono::system_clock::to_time_t(sctp);
				int first = 0;
				double last = 0.0;
				parseFilename(path.filename().string(), first, last);
				info.firstNum = first;
				info.lastNum = last;
				files.push_back(info);
			}
		}
	}
	std::sort(files.begin(), files.end(),
		[](const FileInfo& a, const FileInfo& b)
		{
			return a.firstNum < b.firstNum;
		});

	for (const auto& f : files)
	{
		cv::Mat img = cv::imread(f.path.string(), -1);
		imgVec.push_back(img);
		zVec.push_back(f.lastNum);
		//std::cout << f.path << " | "
		//	<< "first=" << f.firstNum
		//	<< ", last=" << f.lastNum << std::endl;
	}

	return true;


}


int MLIQMetrics::IQMetricUtl::getBinNum(cv::Size s)
{
	int num = round(sqrt((11264 * 9200.0 / (s.area()))));
	return num;
}

double MLIQMetrics::IQMetricUtl::getPix2Arcmin(cv::Size s)
{
	int binNum = getBinNum(s);
	double pixel = IQMetricsParameters::pixel_size;
	double focallength = IQMetricsParameters::FocalLength;
	double pixelPerDeg = atan(pixel * binNum / focallength) * 180.0 / CV_PI * 60;
	return pixelPerDeg;
}

double MLIQMetrics::IQMetricUtl::getPix2Degree(cv::Size s)
{
	int binNum = getBinNum(s);
	double pixel = IQMetricsParameters::pixel_size;
	double focallength = IQMetricsParameters::FocalLength;
	double pixelPerDeg = atan(pixel * binNum / focallength) * 180.0 / CV_PI;
	return pixelPerDeg;
}

void IQMetricUtl::loadJsonConfig(const char* path)
{
	std::ifstream jsonFile(path);
	if (jsonFile.is_open())
	{
		std::string contents =
			std::string((std::istreambuf_iterator<char>(jsonFile)), (std::istreambuf_iterator<char>()));
		jsonFile.close();
		Json settingJsonObj = Json::parse(contents);
		{
			Json& systemJson = settingJsonObj["system"];
			IQMetricsParameters::pixel_size = systemJson["pixel_size"].get<double>();
			IQMetricsParameters::FocalLength = systemJson["FocalLength"].get<double>();
		}

	}
}

void MLIQMetrics::IQMetricUtl::parseFilename(const std::string& name, int& first, double& last)
{

	std::regex re(R"(([-\d\.]+))");  // 提取所有数字（含负号和小数）
	std::sregex_iterator it(name.begin(), name.end(), re);
	std::sregex_iterator end;

	std::vector<std::string> nums;
	for (; it != end; ++it)
	{
		nums.push_back(it->str());
	}

	if (nums.size() >= 2)
	{
		first = std::stoi(nums.front());
		last = std::stod(nums.back());
	}

}
