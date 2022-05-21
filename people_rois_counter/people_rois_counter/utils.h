#pragma once
#define UTILS_H
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class Utils
{
public:
	void add_rois_frame(Mat frame, vector<vector<Point>> ROIS);
	string frame_num_correction(string frame_num);
	bool in_current_region(vector<Point>ROI, Rect box);
	void highlight_ppl(Mat frame,Rect box);
	void count_ppl_certain_region(Mat frame, vector<vector<Point>>ROIS, vector<int>classesId, vector<float>confidence, vector<Rect>boxes);
};

