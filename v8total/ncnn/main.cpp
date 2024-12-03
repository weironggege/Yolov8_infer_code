#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <layer.h>
#include <net.h>




class YOLOV8NCNN
{
public:
	YOLOV8NCNN(float confThreshold, float nmsThreshold);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(ncnn::Net &net, cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	int numproposal;
	int nout;

	float confThreshold;
	float nmsThreshold;

	std::vector<std::string> class_names;

};


YOLOV8NCNN::YOLOV8NCNN(float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;
	this->numproposal = 8400;
	this->nout = 7;

	std::ifstream ifs("classes.txt");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

	// this->module_.load_param(parampath);
	// this->module_.load_model(binpath);
}


void YOLOV8NCNN::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);

}

cv::Mat YOLOV8NCNN::resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;

	int srch = img.rows, srcw = img.cols;
	cv::Mat timg;

	if(srch != srcw)
	{
		float scale_hw = (float)srch / srcw;
		if(scale_hw > 1.0)
		{
			*neww = int(this->inpWidth / scale_hw);
			cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(timg, timg, 0, 0, *padw, this->inpWidth-*neww-*padw, cv::BORDER_CONSTANT, 114);
		
		}
		else
		{
			*newh = int(this->inpHeight * scale_hw);
			cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padh = int((this->inpHeight - *newh) * 0.5);
			cv::copyMakeBorder(timg, timg, *padh, this->inpHeight-*newh-*padh, 0, 0, cv::BORDER_CONSTANT, 114);
		
		}
	
	}
	else
	{
		cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	
	return timg;
}

void YOLOV8NCNN::detect(ncnn::Net &net, cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat simg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	int w = frame.cols;
	int h = frame.rows;

	ncnn::Mat img_in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, neww, newh);

	ncnn::Mat img_ou;
	ncnn::copy_make_border(img_in, img_ou, padh, this->inpHeight-newh-padh, padw, this->inpWidth-neww-padw, ncnn::BORDER_CONSTANT, 114.f);

	const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
	img_ou.substract_mean_normalize(0, norm_vals);
	
	ncnn::Extractor ex = net.create_extractor();

	ex.input("in0", img_ou);

	ncnn::Mat out;
	ex.extract("out0", out);

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;

	cv::Mat rawData = cv::Mat(out.h, out.w, CV_32F, (float*)out.data).t();
	float* pdata = (float*)rawData.data;

	for(int n = 0; n < this->numproposal; ++n)
	{
		float maxss = 0.0;
		int idp = 0;
		for(int k = 0; k < this->nout - 4; ++k)
		{
			if(pdata[k + 4] > maxss)
			{
				maxss = pdata[k + 4];
				idp = k;
			}
		
		}
		if(maxss >= this->confThreshold)
		{
			float cx = (pdata[0] - padw) * ratiow;
			float cy = (pdata[1] - padh) * ratioh;
			float w = pdata[2] * ratiow;
			float h = pdata[3] * ratioh;

			int left = int(cx - 0.5 * w);
			int top = int(cy - 0.5 * h);

			boxes.push_back(cv::Rect(left, top, int(w), int(h)));
			confs.push_back(maxss);
			ids.push_back(idp);
		}
		pdata += this->nout;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);

	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);

}

int main()
{
	YOLOV8NCNN yolo(0.5, 0.5);
	ncnn::Net net;
	net.load_param("v8safehat.param");
	net.load_model("v8safehat.bin");

	cv::Mat srcimg = cv::imread("imgs/te_safehat.jpg");

	yolo.detect(net, srcimg);

	cv::imwrite("imgs/ncnn_cpp_safehat.jpg", srcimg);

	return 0;


}




