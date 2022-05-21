#include "utils.h"

void Utils::add_rois_frame(Mat frame, vector<vector<Point>> ROIS) {
	for (int i = 0; i < ROIS.size(); i++) {
		rectangle(frame, ROIS[i][0], ROIS[i][1], (0, 0, 255), 5);
	}
	return;
}

string Utils::frame_num_correction(string frame_num) {
	int num_added_zeros = 0;
	string zeros = "";
	int next_frame_num = atoi(frame_num.c_str()) + 1;
	if (next_frame_num <= 9)
		num_added_zeros = 3;
	else if (next_frame_num <= 99)
		num_added_zeros = 2;
	else if (next_frame_num <= 999)
		num_added_zeros = 1;

	while (num_added_zeros--) {
		zeros += '0';
	}
	return zeros + to_string(next_frame_num);
}

bool Utils::in_current_region(vector<Point>ROI, Rect box) {

	Rect ROI_rect(ROI[0], ROI[1]);
	return ((ROI_rect & box).area() > 0);
}

void Utils::highlight_ppl(Mat frame,Rect box) {
	int box_x = box.x;
	int box_y = box.y;
	int box_width = box.width;
	int box_height = box.height;
	rectangle(frame, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 255, 255), 2);
}

void Utils::count_ppl_certain_region(Mat frame,vector<vector<Point>>ROIS,vector<int>classesId,vector<float>confidence,vector<Rect>boxes) {
	for (int j = 0; j < ROIS.size(); j++) {
		int curr_ppl_region = 0;
		for (int i = 0; i < classesId.size(); i++) {
			if (classesId[i] != 15)
				continue;
			bool check_existance = in_current_region(ROIS[j], boxes[i]);
			curr_ppl_region = check_existance ? curr_ppl_region += 1 : curr_ppl_region;
			if (confidence[i] >= 0.5) {
				highlight_ppl(frame, boxes[i]);
			}
		}
		putText(frame, "PPl: " + to_string(curr_ppl_region), Point(ROIS[j][0].x, ROIS[j][1].y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
}

