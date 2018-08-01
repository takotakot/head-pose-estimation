#ifndef FACE_CROPPER_H_
#define FACE_CROPPER_H_

#include <vector>
#ifdef DEBUG
#include <cstdio>
#include <sstream>
#endif // DEBUG
#include <iterator>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

typedef cv::Point2f type_point;

cv::RotatedRect RotatedRect_pt(const cv::Point2f &_point1, const cv::Point2f &_point2, const cv::Point2f &_point3);

struct face_metrics
{
    const int chin_index = 8;
    cv::Mat uc, c0, c1, ue, mo2;
    // From AIST人体寸法データベース
    const double length_4 = 231.9;
    const double length_16 = 121.1;
    const double length_8 = 160.8;
    const double length_11 = 147.7;
    double l16, l11;
    double l4, l4mod, l8;

    face_metrics(dlib::full_object_detection &shape);
    cv::Mat get_points_index(int index[], const int &n_index, dlib::full_object_detection &shape);
    cv::Mat get_center_points(dlib::full_object_detection &shape);
    cv::Mat get_eye_points(dlib::full_object_detection &shape);
    cv::Mat get_chin(dlib::full_object_detection &shape);
    std::vector<type_point> get_face_rect();
    std::vector<type_point> get_crop_rect();
    void dump_metric(std::ostream &os);
};

class face_cropper
{
    dlib::frontal_face_detector detector;
    dlib::shape_predictor predictor;
    std::vector<dlib::rectangle> faces;
    std::vector<dlib::full_object_detection> shapes;
    std::vector<face_metrics> metrics;
    const int n_landmarks = 68;

  public:
    face_cropper();
    template <typename T>
    void detect(dlib::cv_image<T> &image);
    int get_num_faces();
    void crop_rotatedrect(cv::Mat &i_img, cv::RotatedRect &rect, cv::Mat &o_img);
    void crop_nth(cv::Mat &i_img, int n, cv::Mat &o_img);
    void dump_metric(int n, std::ostream &os);
};

#endif // FACE_CROPPER_HPP
