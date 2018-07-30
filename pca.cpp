#include<iostream>
#include<vector>
#include<fstream>


// #include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>

//using namespace dlib;
//using namespace std;

const int N_POINTS = 68;
const int N_COL_POINTS = 10;
const int N_ROW_POINTS = 13;

double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D1[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

cv::RotatedRect RotatedRect_pt(const cv::Point2f &_point1, const cv::Point2f &_point2, const cv::Point2f &_point3)
{
	cv::Point2f _center = 0.5f * (_point1 + _point3);
	cv::Vec2f vecs[2];
	vecs[0] = cv::Vec2f(_point1 - _point2);
	vecs[1] = cv::Vec2f(_point2 - _point3);
	// check that given sides are perpendicular
	// CV_Assert(abs(vecs[0].dot(vecs[1])) / (norm(vecs[0]) * norm(vecs[1])) <= FLT_EPSILON);

	// wd_i stores which vector (0,1) or (1,2) will make the width
	// One of them will definitely have slope within -1 to 1
	int wd_i = 0;
	if (std::abs(vecs[1][1]) < std::abs(vecs[1][0]))
		wd_i = 1;
	int ht_i = (wd_i + 1) % 2;

	float _angle = std::atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (float)CV_PI;
	float _width = (float)cv::norm(vecs[wd_i]);
	float _height = (float)cv::norm(vecs[ht_i]);

	return cv::RotatedRect(_center, cv::Size2f(_width, _height), _angle);
}

int main(int argc, char* argv[]) {

	std::ifstream ifs;

	if (argc < 2) {
		ifs.open("4899_points.txt");
	} else {
		ifs.open(argv[1]);
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
	}
	ifs.close();
	// std::cerr << points68_mat << std::endl

	int col_index[] = {27, 28, 29, 30, 33, 51, 62, 66, 57, 8};
	cv::Mat col10(cv::Size(2, N_COL_POINTS), CV_64FC1);

	for(int i = 0; i < N_COL_POINTS; ++i) {
		col10.at<double>(i, 0) = points68_mat.at<double>(col_index[i], 0);
		col10.at<double>(i, 1) = points68_mat.at<double>(col_index[i], 1);
	}
	std::cerr << col10 << std::endl;


	// int row_index[] = {0, 36, 37, 38, 39, 40, 41, 27, 42, 43, 44, 45, 46, 47, 16};
	int row_index[] = {36, 37, 38, 39, 40, 41, 27, 42, 43, 44, 45, 46, 47};
	cv::Mat row13(cv::Size(2, N_ROW_POINTS), CV_64FC1);

	for(int i = 0; i < N_ROW_POINTS; ++i) {
		row13.at<double>(i, 0) = points68_mat.at<double>(row_index[i], 0);
		row13.at<double>(i, 1) = points68_mat.at<double>(row_index[i], 1);
	}
	std::cerr << row13 << std::endl;


	cv::Mat temp;
    if (argc < 2) {
    	temp = cv::imread("IMG_4899.jpg");
    }else{
    	temp = cv::imread(argv[1]);
    }

	/*
	 camera matrix
	 */
	double focal_length = temp.cols;
	cv::Point2d center = cv::Point2d(temp.cols / 2, temp.rows / 2);

	std::cerr << __LINE__ << std::endl;
	// Detect faces
//	std::vector<rectangle> faces = detector(cimg);

	std::ostringstream outtext;
	std::cerr << __LINE__ << std::endl;
	// Find the pose of each face
//	if (faces.size() > 0) {
		//track features
		//full_object_detection shape = predictor(cimg, faces[0]);

		//draw features
		std::cerr << "points" << std::endl;
		for (unsigned int i = 0; i < 68; ++i) {
			//circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 2,
			//		cv::Scalar(0, 0, 255), -1);
			circle(temp, points68[i], 5,
								cv::Scalar(0, 0, 255), -1);
		}

		//fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/

	//}

	cv::PCA pca(points68_mat, cv::Mat(), CV_PCA_DATA_AS_ROW, 2);
	std::cerr << pca.eigenvectors.row(0) << pca.eigenvectors.row(1) << std::endl;

	cv::PCA pca_col(col10, cv::Mat(), CV_PCA_DATA_AS_ROW, 2);
	std::cerr << "col10 eig" << std::endl;
	std::cerr << pca_col.eigenvalues << std::endl;
	std::cerr << pca_col.eigenvectors.row(0) << pca_col.eigenvectors.row(1) << std::endl;
	cv::Mat vf1 = pca_col.eigenvectors.row(0);
	cv::Mat vf2 = pca_col.eigenvectors.row(1);

	std::cerr << "vf" << std::endl;
	std::cerr << vf1 << std::endl;
	std::cerr << vf2 << std::endl;

	cv::Mat uf;
	cv::reduce(col10, uf, 0, CV_REDUCE_AVG);
	std::cerr << "uf" << std::endl;
	std::cerr << uf << std::endl;


	cv::PCA pca_row(row13, cv::Mat(), CV_PCA_DATA_AS_ROW, 2);
	std::cerr << "row13 eig" << std::endl;
	std::cerr << pca_row.eigenvalues << std::endl;
	std::cerr << pca_row.eigenvectors.row(0) << pca_row.eigenvectors.row(1) << std::endl;

	cv::Mat ue;
	cv::reduce(row13, ue, 0, CV_REDUCE_AVG);
	// cv::Vec2d ue = _ue;
	std::cerr << ue << std::endl;

	std::cerr << (ue - uf)*vf1.t() << std::endl;
	std::cerr << (ue - uf)*vf1.t() * vf1 << std::endl;
	cv::Mat mo2 = (ue - uf)*vf1.t() * vf1 + uf;
	std::cerr << mo2 << std::endl;


	cv::Point o2(mo2);
	circle(temp, o2, 5, cv::Scalar(255, 0, 0), -1);

	cv::Point x8(points68_mat.at<double>(8, 0), points68_mat.at<double>(8, 1));
	cv::Mat mx8(points68_mat.row(8));

	double l16 = vf1.dot(mx8 - mo2);
	std::cerr << l16 << std::endl;

	double L4 = 231.9;
	double L16 = 121.1;
	double L8 = 160.8;

	double l4 = l16 * L4 / L16;
	double l4mod = l4 * 1.1;
	double l8 = l16 * L8 / L16;

	// http://tessy.org/wiki/index.php?%B9%D4%CE%F3%A4%CE%BE%E8%BB%BB%A1%A4%C6%E2%C0%D1%A1%A4%B3%B0%C0%D1
	// https://qiita.com/fukushima1981/items/d283b3af3e21d94550c4

	cv::Mat tmp;
	cv::Point A1, B1, C1, D1;

	tmp = mo2 - (l4mod - l16) * vf1 - vf2 * (l8 / 2);
	A1 = cv::Point(tmp);

	tmp += l4mod * vf1;
	B1 = cv::Point(tmp);

	tmp += l8 * vf2;
	C1 = cv::Point(tmp);

	tmp += -l4mod * vf1;
	D1 = cv::Point(tmp);

	/*
	circle(temp, A1, 5, cv::Scalar(255, 0, 0), -1);
	circle(temp, B1, 5, cv::Scalar(255, 0, 0), -1);
	circle(temp, C1, 5, cv::Scalar(255, 0, 0), -1);
	circle(temp, D1, 5, cv::Scalar(255, 0, 0), -1);
	*/
	line(temp, A1, B1, cv::Scalar(0, 255, 0));
	line(temp, B1, C1, cv::Scalar(0, 255, 0));
	line(temp, C1, D1, cv::Scalar(0, 255, 0));
	line(temp, D1, A1, cv::Scalar(0, 255, 0));

	const double m1 = 40, m2 = 30, m3 = 3.5, m4 = 30;

	//https://qiita.com/ChaoticActivity/items/68f10d7452680fa1d52d
	tmp = mo2 - (l4mod + l4mod * m3 / m4 - l16) * vf1 - l4mod * m2 / m4 / 2 * vf2;
	cv::Point2f A(tmp);

	tmp += (l4mod * m1 / m4) * vf1;
	cv::Point2f B(tmp);

	tmp += (l4mod * m2 / m4) * vf2;
	cv::Point2f C(tmp);

	tmp += -(l4mod * m1 / m4) * vf1;
	cv::Point2f D(tmp);

	line(temp, A, B, cv::Scalar(0, 255, 0));
	line(temp, B, C, cv::Scalar(0, 255, 0));
	line(temp, C, D, cv::Scalar(0, 255, 0));
	line(temp, D, A, cv::Scalar(0, 255, 0));

	// clip rotated rect
	// https://qiita.com/vs4sh/items/93d65468a992af5b8f92

// cv3
	// cv::RotatedRect rect(A, B, C); // 回転矩形
	cv::RotatedRect rect;
	rect = RotatedRect_pt(A, B, C);
	cv::Mat M;														  // 回転行列
	cv::Mat rotated;												  // 回転された元画像
	cv::Mat cropped;												  // 切り出された画像

	float angle = rect.angle;
	cv::Size rect_size = rect.size;
	if (rect.angle < -45.)
	{
		angle += 90.0;
		std::swap(rect_size.width, rect_size.height);
	}

	// 回転矩形の角度から回転行列を計算する．
	M = cv::getRotationMatrix2D(rect.center, angle, 1.0);
	// 元画像を回転させる．
	cv::warpAffine(temp, rotated, M, temp.size(), cv::INTER_CUBIC);
	// 回転した画像から矩形領域を切り出す．
	cv::getRectSubPix(rotated, rect_size, rect.center, cropped);

	cv::imwrite("result.jpg", temp);
	cv::imwrite("rotated_crop.jpg", cropped);

	//	std::cerr << outtext.str();

	return 0;
}

