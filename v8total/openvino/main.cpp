#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>


class YOLOV8OPENVINO
{
public:
	YOLOV8OPENVINO(std::string modelxml, float confThreshold, float nmsThreshold);
	void drawPred(cv::Mat& frame, cv::Rect box, float conf, int classid);
	cv::Mat resize_img(cv::Mat& image, int *neww, int*newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	float confThreshold;
	float nmsThreshold;
	
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model_detect;
	
	int numproposal;
	int nout;

	std::vector<std::string> class_names;

};


YOLOV8OPENVINO::YOLOV8OPENVINO(std::string modelxml, float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	
	this->inpWidth = 640;
	this->inpHeight = 640;

	std::ifstream ifs("coco.names");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));
	
	this->nout = 84;
	this->numproposal = 8400;

	ov::Core core;

	this->compiled_model_detect = core.compile_model(modelxml, "CPU");

	this->infer_request = this->compiled_model_detect.create_infer_request();

}


void YOLOV8OPENVINO::drawPred(cv::Mat& frame, cv::Rect box, float conf, int classid)
{
	cv::rectangle(frame, box, cv::Scalar(123,45,67), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(90,123,89), 2);

}

cv::Mat YOLOV8OPENVINO::resize_img(cv::Mat& image, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;

	int srcw = image.cols, srch = image.rows;

	cv::Mat simg;

	if(srcw != srch)
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

void YOLOV8OPENVINO::detect(cv::Mat& frame)
{
	int neww =0, newh = 0, padw = 0, padh = 0;
	cv::Mat simg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	cv::Mat blob = cv::dnn::blobFromImage(simg, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	auto input_port = this->compiled_model_detect.input();
	
	ov::Tensor inputTensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));	

	
	this->infer_request.set_input_tensor(inputTensor);
	

	this->infer_request.infer();

	auto output = infer_request.get_output_tensor(0);
	float *outdata = output.data<float>();
	cv::Mat output_buffer(this->nout, this->numproposal, CV_32F, outdata);
	cv::transpose(output_buffer, output_buffer);
	float* pdata = (float*)output_buffer.data;

	
	std::vector<cv::Rect> boxes;
	std::vector<int> ids;
	std::vector<float> confs;

	float ratioh = (float)frame.rows / newh;
	float ratiow = (float)frame.cols / neww;

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

	for(int idx : indices) this->drawPred(frame,  boxes[idx], confs[idx], ids[idx]);

}


int main()
{
	YOLOV8OPENVINO net("yolov8n_openvino_model/yolov8n.xml", 0.7, 0.8);

	cv::Mat srcimg = cv::imread("imgs/bus.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/bus_openvino_cpp.jpg", srcimg);

	return 0;
}








