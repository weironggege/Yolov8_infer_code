#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sophon/sail/api.h>

class YOLOV8BM
{
public:
	YOLOV8BM(std::string modelpath, float confThreshold, float nmsThreshold);
	void drawPred(cv::Mat& frame, cv::Rect box, float conf, int clsid);
	cv::Mat resize_img(cv::Mat image, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& srcimg);
private:
	float confThreshold;
	float nmsThreshold;

	int nout;
	int numproposal;

	int inpWidth;
	int inpHeight;
	
	std::vector<std::string> class_names;

	int tpuid;

	sophon::SAIL_API_HANDLE handle;
	sophon::SAIL_API_MODEL model;
	sophon::SAIL_API_ENGINE engine;
	


};



YOLOV8BM::YOLOV8BM(std::string modelpath, float confThreshold, float nmsThreshold)
{

	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;

	this->numproposal = 8400;
	this->nout = 84;

	this->tpuid = 0;

	std::iftream ifs("coco.names");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.lengt()-1));
	
	this->handle = sophon::create_sail_api_handle();

	this->model = sophon::load_model(this->handle, modelpath);

	this->engine = sophon::create_engine(this->model, this->tpuid);
}

void YOLOV8BM::drawPred(cv::Mat& frame, cv::Rect box, float conf, int clsid)
{
	cv::rectangle(frane, box, cv::Scalar(89,45,234), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[clsid] + ":" + label;
	
	cv::putText(frame, label, cv::Point(box.x,box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(89,78,123), 2);
}

cv::Mat YOLOV8BM::resize_img(cv::Mat image, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;

	int srch = image.rows, srcw = image.cols;
	cv::Mat simg;
	if(srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if(hw_scale > 1.0)
		{
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(image, simg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(simg, simg, 0, 0, *padw, this->inpWidth-*neww-*padw, cv::BORDER_CONSTANT, 114);
		}
		else
		{
			*newh = int(this->inpHeight * hw_scale);
			cv::resize(image, simg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padh = int((this->inpHeight - *newh) * 0.5);
			cv::copyMakeBorder(simg, simg, *padh, this->inpHeight-*newh-*padh, 0, 0, cv::BORDER_CONSTANT, 114);
		}
	
	}
	else
	{
		cv::resize(image, simg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return simg;

}

void YOLOV8BM::detect(cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat oimg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	cv::Mat blob = cv::dnn::blobFromImage(oimg, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	std::vector<sophon::SAIL_TENSOR> inputs;
	sophon::SAIL_TENSOR input_tensor;

	input_tensor.shape = {1, 3, this->inpHeight, this->inpWidth};
	input_tensor.dtype = sophon::SAIL_FLOAT32;
	input_tensor.scale = 1.0f;
	input_tensor.data = blob.data;

	inputs.push_back(input_tensor);

	std::vector<sophon::SAIL_TENSOR> outputs;
	sophon::predict(this->engine, inputs, &outputs);

	float* out = static_cast<float*>(outputs[0].data);
	cv::Mat output_buffer(this->nout, this->numproposal, CV_32F, out);
	cv::transpose(output_buffer, output_buffer);
	float* pdata = (float*)output_buffer.data;

	std::vector<cv::Rect> boxes;
	std::vector<float> confs;
	std::vector<int> ids;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;

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
	for(int idx : indices) this->drawPred(frame, boxes[idx], confs[idx], ids[idx]);


}


int main()
{
	YOLOV8BM v8bm("weights/yolov8n_fp32.bmodel");

	cv::Mat srcimg = cv::imread("imgs/bus.jpg");

	v8bm.detect(srcimg);

	cv::imwrite("imgs/bus_bm_cpp.jpg", srcimg);

	return 0;
}












