#define img_width 750
#define img_height 700
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "utils.h"

using namespace cv;
using namespace std;

int main() {
	Utils utils;
	string frame_num = "0000";
	string pre_trained_model = "E:\\A-eye-tech-task\\pre_trained_models\\MobileNetSSD_deploy.caffemodel";
	string pre_trained_config = "E:\\A-eye-tech-task\\pre_trained_models\\MobileNetSSD_deploy.prototxt.txt";
	
	vector<string> classes = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
			"diningtable",  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
	
	vector<vector<Point>> ROIS = { {Point(10,8),Point(747,566)},{Point(287,156),Point(711,431)},{Point(27,129),Point(230,289)} };
	auto model = dnn::DetectionModel(pre_trained_model, pre_trained_config);
	model.setInputSize(Size(img_width, img_height));
	model.setInputMean(Scalar(127.5, 127.5, 127.5));
	model.setInputScale(1.0 / 127.5);
	while (true) {
		string data_path = "E:\\A-eye-tech-task\\Crowd_PETS09\\S1\\L1\\Time_13-57\\View_001\\frame_" + frame_num + ".jpg";
		Mat image = imread(data_path, IMREAD_COLOR);
		if (image.data == NULL)
			break;

		utils.add_rois_frame(image, ROIS);
		frame_num = utils.frame_num_correction(frame_num);



		vector<int>classesId;
		vector<float>confidence;
		vector<Rect>boxes;

		
		model.detect(image, classesId, confidence,boxes,0.5,0.5);

		utils.count_ppl_certain_region(image,ROIS,classesId,confidence,boxes);



		imshow("Output Image", image);
		int key = waitKey(1);
		if (key == 'q')
			return 0;
	}
	
	destroyAllWindows();

	return 0;
}