#include<iostream>
#include<vector>
#include<fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/*
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
*/

const int N_POINTS = 68;
const int N_COL_POINTS = 10;


/*
// under construction
struct calc_shape_rectangle{
	dlib::full_object_detection shape;
	cv::Mat v_pca;
	std::vector<int> col_index;
	cv::Point pts_mean, pts_eye_mean, pt_eye_cross, pt_chin;

	calc_shape_rectangle(dlib::full_object_detection shape) {
		this->v_pca = cv::Mat(2, 2, CV_64FC1);
		int col_index[] = {27, 28, 29, 30, 33, 51, 62, 66, 57, 8};
		for(int i = 0; i < sizeof(col_index)/sizeof(col_index[0]); ++i) {
			this->col_index.push_back(col_index[i]);
		}
		this->shape = shape;
	}

	void pca() {
		cv::Mat points68_mat(cv::Size(2, N_POINTS), CV_64FC1);


		return;
	}

};
*/



/**
 * https://blanktar.jp/blog/2015/07/python-opencv-crop-box.html
 *
 * 	trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
	return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。
 *
 *https://iwaki2009.blogspot.jp/2013/01/opencv_30.html
 *OpenCVで傾いた矩形を描画する

FloodFillなどの処理により、マスク画像を取得した後に、大まかな領域情報を得る手法の一つとして、cv::findContoursで輪郭線を検出し、さらにcv::minAreaRectで全体を囲んだ四角形（傾きあり）を使用することがある。

ただし、cv::minAreaRectの戻り値が、cv::RotatedRectであるため、cv::rectangleでは、描画できない。

そのため、cv::RotatedRectから、４点を取り出し、自分で４つの線を描画する必要がある

cv::RotatedRect rRect =  cv::minAreaRect(polygon);

cv::Point2f vertices[4];
rRect.points(vertices);
for (int i = 0; i < 4; i++)
 line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));

説明はここ
 *
 */

// circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
void write_circle(cv::Mat &img, cv::Point pt, cv::Scalar col) {
	cv::circle(img, pt, 10, col, -1);
	return;
}

int main(int argc, char* argv[])
{
	cv::Scalar col_red(0, 0, 255);
    // Grab a frame
    cv::Mat img;
    // cap >> temp;
    if (argc < 2) {
    	img = cv::imread("IMG_4899.jpg");
    }else{
    	img = cv::imread(argv[1]);
    }

    std::ifstream ifs;
	if (argc < 3) {
		ifs.open("4899_points.txt");
	} else {
		ifs.open(argv[2]);
	}

	int x, y;
	std::vector<cv::Point2d> points68;
	cv::Mat points68_mat(cv::Size(2, N_POINTS), CV_64FC1);

	for (int i = 0; i < N_POINTS; ++i) {
		ifs >> x;
		ifs >> y;
		// std::cerr << x << "\t" << y << std::endl;
		points68.push_back(cv::Point2d(x, y));
		points68_mat.at<double>(i, 0) = x;
		points68_mat.at<double>(i, 1) = y;
		//points68_mat.at(i) = cv::Point2d(x, y);

		// std::cerr << x << y << std::endl;
	}
	ifs.close();
	// std::cerr << points68_mat << std::endl

	int col_index[] = {27, 28, 29, 30, 33, 51, 62, 66, 57, 8};
	cv::Mat col10(cv::Size(2, N_COL_POINTS), CV_64FC1);

	for(int i = 0; i < N_COL_POINTS; ++i) {
		col10.at<double>(i, 0) = points68_mat.at<double>(col_index[i], 0);
		col10.at<double>(i, 1) = points68_mat.at<double>(col_index[i], 1);
		write_circle(img, cv::Point(points68_mat.at<double>(col_index[i], 0), points68_mat.at<double>(col_index[i], 1)), col_red);
		// circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
	}
	// std::cerr << col10 << std::endl;



//    write_circle(img, pt1, col_red);
    cv::imwrite("result.jpg", img);

	return 0;
}
