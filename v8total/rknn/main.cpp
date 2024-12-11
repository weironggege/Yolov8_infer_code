#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"



class YOLOV8RKNN
{
public:
	YOLOV8RKNN(float confThreshold, float nmsThreshold);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	cv::Mat resize_img(cv::Mat image, int *neww, int *newh, int *padw, int *padh);
	void normalize_(cv::Mat img);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	float confThreshold;
	float nmsThreshold;

	int numproposal;
	int nout;

	std::vector<std::string> class_names;

	std::vector<int> input_img;


};


YOLOV8RKNN::YOLOV8RKNN(float confThreshold, float nmsThreshold)
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
		
}


void YOLOV8RKNN::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(90, 78, 231), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(89, 233, 0), 2);

}

cv::Mat YOLOV8RKNN::resize_img(cv::Mat image, int *neww, int *newh, int *padw, int *padh)
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

void YOLOV8RKNN::normalize_(cv::Mat img)
{
        int srch = img.rows, srcw = img.cols;
        this->input_img.resize(srch * srcw * img.channels());

        for(int c = 0; c < 3; ++c)
        {
                for(int i = 0; i < srch; ++i)
                {

                        for(int j = 0; j < srcw; ++j)
                        {
                                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];

                                this->input_img[c * srcw * srch + i *srcw + j] = int(pix);

                        }

                }

        }


}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}


float iou(const cv::Rect& box1, const cv::Rect& box2) {
    float inter_xmin = std::max(box1.x, box2.x);
    float inter_ymin = std::max(box1.y, box2.y);
    float inter_xmax = std::min(box1.x + box1.width, box2.x + box2.width);
    float inter_ymax = std::min(box1.y + box1.height, box2.y + box2.height);

    float inter_width = std::max(0.0f, inter_xmax - inter_xmin);
    float inter_height = std::max(0.0f, inter_ymax - inter_ymin);
    float inter_area = inter_width * inter_height;

    float box1_area = box1.width * box1.height;
    float box2_area = box2.width * box2.height;
    float union_area = box1_area + box2_area - inter_area;

    return inter_area / union_area;
}

// 自定义NMS函数
void customNMSBoxes(const std::vector<cv::Rect>& boxes, 
                    const std::vector<float>& confs,
                    float confThreshold,
                    float nmsThreshold,
                    std::vector<int>& indices) {
    
    // 过滤掉置信度低于阈值的框
    std::vector<std::pair<int, float>> filtered_boxes;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (confs[i] >= confThreshold) {
            filtered_boxes.emplace_back(i, confs[i]);
        }
    }

    // 按照置信度从高到低排序
    std::sort(filtered_boxes.begin(), filtered_boxes.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    indices.clear();
    while (!filtered_boxes.empty()) {
        // 选择当前具有最高置信度的框索引
        int best_box_index = filtered_boxes.front().first;
        indices.push_back(best_box_index);

        // 过滤掉与best_box IoU大于nmsThreshold的其他框
        auto it = filtered_boxes.begin() + 1;
        while (it != filtered_boxes.end()) {
            int other_box_index = it->first;
            if (iou(boxes[best_box_index], boxes[other_box_index]) >= nmsThreshold) {
                it = filtered_boxes.erase(it); // 删除满足条件的框
            } else {
                ++it;
            }
        }
        filtered_boxes.erase(filtered_boxes.begin()); // 移除已经处理过的最高置信度框
    }
}



void YOLOV8RKNN::detect(cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat oimg = this->resize_img(frame, &neww, &newh, &padw, &padh);

	// this->normalize_(oimg);
	// cv::Mat srcimg;
	// cv::cvtColor(oimg, srcimg, cv::COLOR_BGR2RGB);
	oimg.convertTo(oimg, CV_32F, 1.0);

	int model_data_size = 0;
	const char* model_path = "yolov8n.rknn";
	unsigned char* model_data = load_model(model_path, &model_data_size);
	rknn_context ctx;

	rknn_init(&ctx, model_data, model_data_size, 0, NULL);

	rknn_sdk_version version;
	rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
	

	
	rknn_input inputs[1];
    	inputs[0].size = this->inpHeight * this->inpWidth * 3 * sizeof(float);
    	inputs[0].index = 0;
    	inputs[0].pass_through = 0;
    	inputs[0].type = RKNN_TENSOR_FLOAT32;
    	inputs[0].fmt = RKNN_TENSOR_NHWC;
	
	inputs[0].buf = (float *)oimg.data;
	

	rknn_inputs_set(ctx, 1, inputs);
	
	rknn_run(ctx, NULL);
	
	
	rknn_output outputs[1];

	outputs[0].want_float = 1;
	rknn_outputs_get(ctx, 1, outputs, NULL);
	
		
	float* outdata = (float*)outputs[0].buf;
	
	std::vector<float> prs;
	
       	for(int i = 0; i < this->numproposal; ++i)
	{
		for(int j = 0; j < this->nout; ++j)
		{
			prs.push_back(outdata[j * this->numproposal + i]);
		}
	}	

	float* pdata = (float*)prs.data();
	
	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;
	
	
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

	customNMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);
	
	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);
	

}

int main()
{
	YOLOV8RKNN net(0.7, 0.8);

	cv::Mat srcimg = cv::imread("imgs/bus.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/bus_rt.jpg", srcimg);

	return 0;
}












