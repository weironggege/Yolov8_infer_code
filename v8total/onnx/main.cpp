#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

#include <onnxruntime_cxx_api.h>

class YOLOV8
{
public:
	YOLOV8(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	void normalize_(cv::Mat img);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:
	
	int inpWidth;
	int inpHeight;

	int numproposal;
	int nout;

	float confThreshold;
	float nmsThreshold;

	std::vector<float> input_image_;
        std::vector<std::string> class_names;

	std::vector<char*> input_names;
	std::vector<char*> output_names;

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV8");	
	Ort::Session *sess = nullptr;
	Ort::SessionOptions sessops = Ort::SessionOptions();



};


YOLOV8::YOLOV8(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;

	this->numproposal = 8400;
	this->nout = 84;

	std::ifstream ifs("coco.names");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

	OrtSessionOptionsAppendExecutionProvider_CUDA(this->sessops, 0);
	this->sessops.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	this->sess = new Ort::Session(this->env, modelpath.c_str(), this->sessops);

	size_t innodes = this->sess->GetInputCount();
	size_t ounodes = this->sess->GetOutputCount();

	Ort::AllocatorWithDefaultOptions allocator;

	for(size_t i = 0; i < innodes; ++i) this->input_names.push_back(this->sess->GetInputName(i, allocator));

	for(size_t i = 0; i < ounodes; ++i) this->output_names.push_back(this->sess->GetOutputName(i, allocator));



}


void YOLOV8::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;
	
	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(114,114,0), 2);

}

void YOLOV8::normalize_(cv::Mat img)
{
	int srch = img.rows, srcw = img.cols;
	this->input_image_.resize(srch * srcw * img.channels());

	for(int c = 0; c < 3; ++c)
	{
		for(int i = 0; i < srch; ++i)
		{
		
			for(int j = 0; j < srcw; ++j)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * srcw * srch + i *srcw + j] = pix / 255.0;
				
			}
		
		}
	
	}


}


cv::Mat YOLOV8::resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh)
{

	*neww = this->inpWidth, *newh = this->inpHeight;
	int srch = img.rows, srcw = img.cols;
	cv::Mat timg;
	if(srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if(hw_scale > 1.0)
		{
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(timg, timg, 0, 0, *padw, this->inpWidth-*neww-*padw, cv::BORDER_CONSTANT, 114);
		
		}
		else
		{
			*newh = int(this->inpHeight * hw_scale);
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



void YOLOV8::detect(cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat simg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	this->normalize_(simg);

	std::array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	std::vector<Ort::Value> output_tensors = this->sess->Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor, 1, output_names.data(), output_names.size());

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;
	auto ouptr = output_tensors[0].GetTensorMutableData<float>();

	cv::Mat rawData = cv::Mat(cv::Size(this->numproposal, this->nout), CV_32F, ouptr).t();
	float* pdata = (float*)rawData.data;

	for(int n = 0; n < this->numproposal; ++n)
	{
		float maxss = 0.0;
		int idp = 0;
		for(int k = 0; k < this->nout-4; ++k)
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
	YOLOV8 net(0.7, 0.8, "weights/yolov8n.onnx");

	cv::Mat srcimg = cv::imread("imgs/bus.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/bus_onnx_cpp.jpg", srcimg);
	
	return 0;
}









