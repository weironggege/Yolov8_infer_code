#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

class YOLOV8
{
public:
	YOLOV8(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	float confThreshold;
	float nmsThreshold;

	std::vector<std::string> class_names;

	cv::dnn::Net net;

};

YOLOV8::YOLOV8(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;

	this->net = cv::dnn::readNet(modelpath);

	std::ifstream ifs("coco.names");
	std::string line;
	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

}

void YOLOV8::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);
	
	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);

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
	cv::Mat blob = cv::dnn::blobFromImage(simg, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	this->net.setInput(blob);

	std::vector<cv::Mat> outs;

	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int numproposal = outs[0].size[2];
	int nout = outs[0].size[1];
	outs[0] = outs[0].reshape(0, nout);
	cv::transpose(outs[0], outs[0]);

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;
	float* pdata = (float*)outs[0].data;

	for(int n = 0; n < numproposal; ++n)
	{
		float maxss = 0.0;
		int idp = 0;
		for(int k = 0; k < nout - 4; ++k)
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

		pdata += nout;
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

	cv::imwrite("imgs/bus_tmp_cpp.jpg", srcimg);

	return 0;

}






