#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include"ML_CTF.h"
#include"ML_Uniformity.h"
//#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"
#include"ml_image_public.h"
#include <embed.h>
namespace py = pybind11;
using namespace MLIQMetrics;
using namespace MLImageDetection;
using namespace cv;
using namespace std;
void ctfTest()
{
	cv::Mat img = cv::imread("I:\\IMGS\\JIG\\OBJ_2.6_A00100425-1(4#)\\Grid Contrast Horizontal\\Data\\1_17.36734_-49.74066_14.2014.tif", -1);
	MLCTF ctf;
	string savePath = "I:\\IMGS\\JIG\\OBJ_2.6_A00100425-1(4#)\\Grid Contrast Horizontal\\";
	string path = "I:\\IMGS\\JIG\\OBJ_2.6_A00100425-1(4#)\\Grid Contrast Horizontal\\Data\\";
	//CTFRe re=ctf.getCTF(img, savePath);
	CTFRe re = ctf.getCTF(path, savePath);
}
void plotTest()
{
	Gnuplot gp;
	std::vector<std::pair<double, double>> pts;

	for (double x = -5; x <= 5; x += 0.1) {
		pts.push_back(std::make_pair(x, sin(x)));
	}
	gp << "plot '-' with lines\n";
	gp.send(pts);
	gp << "e\n";
}
void luminanceTest()
{
}
void pythonTest1()
{
	//	py::scoped_interpreter guard{};  // 启动 Python 解释器

		//try {
		//	py::module_ sys = py::module_::import("sys");

		//	sys.attr("path").attr("insert")(0,"E:/project/jig/src/plugins/IQMetricsTest");
		//	// ?? 导入模块（不要写 .py）
		//	py::module_ test = py::module_::import("test");
		//	int result = test.attr("add")(2, 3).cast<int>();
		//	std::cout << "add result = " << result << std::endl;

		//	//std::string msg = test.attr("greet")("Tom").cast<std::string>();
		//	//std::cout << msg << std::endl;

		//}
		//catch (const std::exception& e) {
		//	std::cout << "Error: " << e.what() << std::endl;
		//}



}
void uniformityTest()
{
	MLUniformity uni;
	string path = "I:\\IMGS\\JIG\\OBJ_2.6_A00100425-1(4#)\\Illumination\\Data\\";
	string savepath = "I:\\IMGS\\JIG\\OBJ_2.6_A00100425-1(4#)\\Illumination\\";
	uni.getUniformity(path, savepath);

}
void main()
{
	//uniformityTest();
	//pythonTest1();
	//plotTest();
	ctfTest();
}