#ifndef FACE_CROPPER_H_
#define FACE_CROPPER_H_

#define HEAD_POSE_ESTIMATION_DEBUG 1

#include <vector>
#include <cmath>
#ifdef DEBUG
#include <cstdio>
#include <sstream>
#endif // DEBUG
#include <iterator>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

typedef cv::Point2f type_point;
typedef cv::Matx44d head_pose;

// From: https://github.com/chili-epfl/attention-tracker/
// Anthropometric for male adult
// Relative position of various facial feature relative to sellion
// Values taken from https://en.wikipedia.org/wiki/Human_head
// X points forward
// Original

const static cv::Point3f P3D_SELLION(0., 0., 0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -65.5, -5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 65.5, -5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -77.5, -6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 77.5, -6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -48.0);
const static cv::Point3f P3D_STOMMION(10.0, 0., -75.0);
const static cv::Point3f P3D_MENTON(0., 0., -133.0);

// In mm scale
/*
const static cv::Point3f P3D_SELLION(0., 0., 0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -45.55, -5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 45.55, -5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -74.25, -6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 74.25, -6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -48.0);
const static cv::Point3f P3D_SUBNASALE(0., 0., -48.0);
const static cv::Point3f P3D_STOMION(10.0, 0., -75.0);
const static cv::Point3f P3D_MENTON(-32.14, 0., -116.76);
*/

const static double THETA_0 = M_2_PI - std::acos((121.1 * 121.1 + 53.0 * 53.0 - 71.4 * 71.4) / (2 * 121.1 * 53.0));

// Interesting facial features with their landmark index
enum FACIAL_FEATURE
{
    NOSE = 30,
    SUBNASALE = 33,
    RIGHT_EYE = 36,
    LEFT_EYE = 45,
    RIGHT_SIDE = 0,
    LEFT_SIDE = 16,
    EYEBROW_RIGHT = 21,
    EYEBROW_LEFT = 22,
    MOUTH_UP = 51,
    MOUTH_DOWN = 57,
    MOUTH_RIGHT = 48,
    MOUTH_LEFT = 54,
    SELLION = 27,
    MOUTH_CENTER_TOP = 62,
    MOUTH_CENTER_BOTTOM = 66,
    MENTON = 8
};

cv::RotatedRect RotatedRect_pt(const cv::Point2f &_point1, const cv::Point2f &_point2, const cv::Point2f &_point3);
bool isRotationMatrix(cv::Mat &R);
cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R);

struct face_metrics
{
    const int chin_index = 8;
    cv::Mat uc, c0, c1, ue, mo2;
    cv::Mat mid_side;
    // From AIST人体寸法データベース
    const double length_4 = 231.9;
    const double length_16 = 121.1;
    const double length_16_z = 116.76;
    const double length_8 = 160.8;
    const double length_11 = 147.7;
    double l16, l11;
    double l4, l4mod, l8;
    head_pose pose;
    double roll, pitch, yaw;
    double roll2, pitch2, yaw2;
    double focal_length;
    cv::Point2d center;
#ifdef HEAD_POSE_ESTIMATION_DEBUG
    mutable cv::Mat _debug;
#endif

    face_metrics(double focal_length, cv::Point2d center, dlib::full_object_detection &shape);
    cv::Point2f coordsOf(dlib::full_object_detection &shape, FACIAL_FEATURE feature);
    cv::Mat get_points_index(int index[], const int &n_index, dlib::full_object_detection &shape);
    cv::Mat get_center_points(dlib::full_object_detection &shape);
    cv::Mat get_eye_points(dlib::full_object_detection &shape);
    cv::Mat get_chin(dlib::full_object_detection &shape);
    std::vector<double> calc_euler(dlib::full_object_detection &shape);
    std::vector<type_point> get_face_rect();
    cv::Mat get_crop_upleft();
    std::vector<type_point> get_crop_rect();
    void dump_metric(std::ostream &os);
#ifdef HEAD_POSE_ESTIMATION_DEBUG
    void add_debug_image(cv::Mat &image);
#endif

  private:
    head_pose calc_pose(dlib::full_object_detection &shape);
};

class face_cropper
{
    dlib::frontal_face_detector detector;
    dlib::shape_predictor predictor;
    std::vector<dlib::rectangle> faces;
    std::vector<dlib::full_object_detection> shapes;
    std::vector<face_metrics> metrics;
    const int n_landmarks = 68;
    double angle;

  public:
    face_cropper();
    template <typename T>
    void detect(dlib::cv_image<T> &image);
    int get_num_faces();
    void crop_rotatedrect(cv::Mat &i_img, cv::RotatedRect &rect, cv::Mat &o_img);
    void crop_nth(cv::Mat &i_img, int n, cv::Mat &o_img);
    void dump_metric(int n, std::ostream &os);
};

inline cv::Point2f toCv(const dlib::point &p)
{
    return cv::Point2f(p.x(), p.y());
}

#endif // FACE_CROPPER_HPP
