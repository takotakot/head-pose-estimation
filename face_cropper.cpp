#include "face_cropper.h"
#include "functions.h"
#include <omp.h>

// #define omp_get_thread_num() (0)
#define HAVE_ROTATEDRECT_3PT 0

cv::RotatedRect RotatedRect_pt(const cv::Point2d &_point1, const cv::Point2d &_point2, const cv::Point2d &_point3)
{
    cv::Point2d _center = 0.5f * (_point1 + _point3);
    cv::Vec2d vecs[2];
    vecs[0] = cv::Vec2d(_point1 - _point2);
    vecs[1] = cv::Vec2d(_point2 - _point3);
    // check that given sides are perpendicular
    CV_Assert(abs(vecs[0].dot(vecs[1])) / (norm(vecs[0]) * norm(vecs[1])) <= FLT_EPSILON);

    // wd_i stores which vector (0,1) or (1,2) will make the width
    // One of them will definitely have slope within -1 to 1
    int wd_i = 0;
    if (std::abs(vecs[1][1]) < std::abs(vecs[1][0]))
        wd_i = 1;
    int ht_i = (wd_i + 1) % 2;

    double _angle = std::atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (double)CV_PI;
    double _width = (double)cv::norm(vecs[wd_i]);
    double _height = (double)cv::norm(vecs[ht_i]);

    return cv::RotatedRect(_center, cv::Size2d(_width, _height), _angle);
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return cv::norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R)
{

    assert(isRotationMatrix(R));

    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    double x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

face_metrics::face_metrics(double focal_length, cv::Point2d center, dlib::full_object_detection &shape) : focal_length(focal_length), center(center)
{
    cv::Mat center_points = get_center_points(shape);
    cv::Mat eye_points = get_eye_points(shape);

    cv::reduce(center_points, uc, 0, CV_REDUCE_AVG);
    cv::PCA pca_center_points(center_points, cv::Mat(), CV_PCA_DATA_AS_ROW, 2);
    c0 = pca_center_points.eigenvectors.row(0);
    c1 = pca_center_points.eigenvectors.row(1);
    cv::reduce(eye_points, ue, 0, CV_REDUCE_AVG);

    pose = face_metrics::calc_pose(shape);

    mo2 = (ue - uc) * c0.t() * c0 + uc;

    cv::Mat t_right(cv::Size(2, 1), CV_64FC1), t_left(cv::Size(2, 1), CV_64FC1);
    t_right.at<double>(0, 0) = shape.part(RIGHT_SIDE).x();
    t_right.at<double>(0, 1) = shape.part(RIGHT_SIDE).y();
    t_left.at<double>(0, 0) = shape.part(LEFT_SIDE).x();
    t_left.at<double>(0, 1) = shape.part(LEFT_SIDE).y();

    mid_side = (t_right + t_left) / 2;

    l16 = c0.dot(get_chin(shape) - mo2);
    double dx = shape.part(16).x() - shape.part(0).x(), dy = shape.part(16).y() - shape.part(0).y();
    l11 = std::sqrt(dx * dx + dy * dy);
    l4 = l16 * length_4 / length_16;
    l8 = l16 * length_8 / length_16;
    // heuristic: if l16 is too small compared to l11, set l4mod longer
    // TODO: refine
    double l4mod_coef = 1.15;
    if (pitch2 < 1.416) {
        l4mod_coef += (1.416 - pitch2) * 1;
    }
    l4mod = l4 * l4mod_coef;
    std::cerr << "l16: " << l16 << std::endl;
    // std::cerr << l4 << std::endl;
}

type_point face_metrics::coordsOf(dlib::full_object_detection &shape, FACIAL_FEATURE feature)
{
    return toCv(shape.part(feature));
}

cv::Mat face_metrics::get_points_index(int index[], const int &n_index, dlib::full_object_detection &shape)
{
    cv::Mat point_matrix(cv::Size(2, n_index), CV_64FC1);
    for (int i = 0; i < n_index; ++i)
    {
        point_matrix.at<double>(i, 0) = shape.part(index[i]).x();
        point_matrix.at<double>(i, 1) = shape.part(index[i]).y();
    }

    return point_matrix;
}

cv::Mat face_metrics::get_center_points(dlib::full_object_detection &shape)
{
    // center 10
    int center_points_index[] = {27, 28, 29, 30, 33, 51, 62, 66, 57, 8};
    const int n_center_points = sizeof(center_points_index) / sizeof(center_points_index[0]);

    return get_points_index(center_points_index, n_center_points, shape);
}

cv::Mat face_metrics::get_eye_points(dlib::full_object_detection &shape)
{
    // eye line 13
    int eye_points_index[] = {36, 37, 38, 39, 40, 41, 27, 42, 43, 44, 45, 46, 47};
    const int n_eye_points = sizeof(eye_points_index) / sizeof(eye_points_index[0]);

    return get_points_index(eye_points_index, n_eye_points, shape);
}

cv::Mat face_metrics::get_chin(dlib::full_object_detection &shape)
{
    cv::Mat point_matrix(cv::Size(2, 1), CV_64FC1);
    point_matrix.at<double>(0, 0) = shape.part(chin_index).x();
    point_matrix.at<double>(0, 1) = shape.part(chin_index).y();

    return point_matrix;
}

std::vector<double> calc_euler(dlib::full_object_detection &shape)
{
    return std::vector<double>();
}

// unused
std::vector<type_point> face_metrics::get_face_rect()
{
    cv::Mat tmp;
    std::vector<type_point> rect;

    tmp = mo2 - (l4mod - l16) * c0 - (l8 / 2) * c1;
    rect.push_back(type_point(tmp));

    tmp += l4mod * c0;
    rect.push_back(type_point(tmp));

    tmp += l8 * c1;
    rect.push_back(type_point(tmp));

    tmp += -l4mod * c1;
    rect.push_back(type_point(tmp));

    return rect;
}

cv::Mat face_metrics::get_crop_upleft()
{
    const double m1 = 40, m2 = 30, m3 = 3.5, m4 = 30;
    cv::Mat tmp, diff;
    // TODO: refine
    // Strategy1: if yaw2 is largeer than -1.63 move left
    // less than -1.63 move right
    // Strategy2: move counter ward from the point which is in the middle of ears
    // double yaw2_diff = yaw2 - (-1.63);
    // double y2d_cos = std::cos(yaw2_diff);

    int mode = 2;

    tmp = mo2 - (l4mod + l4mod * m3 / m4 - l16) * c0 - (l4mod * m2 / m4 / 2) * c1;
    if (mode == 2)
    {
        diff = mid_side - uc;
        // std::cerr << "mid_side - uc: " << diff << std::endl;
        tmp += (diff * c1.t() * c1) / 2;
    }

    return tmp;
}

/*
        Refactorization is needed:
        http://tessy.org/wiki/index.php?%B9%D4%CE%F3%A4%CE%BE%E8%BB%BB%A1%A4%C6%E2%C0%D1%A1%A4%B3%B0%C0%D1
        https://qiita.com/fukushima1981/items/d283b3af3e21d94550c4
        https://qiita.com/ChaoticActivity/items/68f10d7452680fa1d52d
    */
std::vector<type_point> face_metrics::get_crop_rect()
{
    const double m1 = 40, m2 = 30, m3 = 3.5, m4 = 30;
    cv::Mat tmp;
    std::vector<type_point> rect;

    tmp = get_crop_upleft();
    rect.push_back(type_point(tmp));

    tmp += (l4mod * m1 / m4) * c0;
    rect.push_back(type_point(tmp));

    tmp += (l4mod * m2 / m4) * c1;
    rect.push_back(type_point(tmp));

    tmp += -(l4mod * m1 / m4) * c0;
    rect.push_back(type_point(tmp));

    return rect;
}

void face_metrics::dump_metric(std::ostream &os)
{
    os << l16 << "\t" << l11 << "\t" << l11 / l16 << "\t";
    os << roll << "\t" << pitch << "\t" << yaw << "\t";

    // roll: large clockwise, small unticlockwise (viewing from object)
    // avg: 0.578
    // pitch: large up,  small down
    // avg: 1.42
    // yaw: large right, small left (viewing from object)
    // avg: -1.63
    // if too big, move left (viewing from camera)
    os << roll2 << "\t" << pitch2 << "\t" << yaw2;
}

void face_metrics::add_debug_image(cv::Mat &image)
{
    _debug = image.clone();
}

head_pose face_metrics::calc_pose(dlib::full_object_detection &shape)
{
    cv::Mat projectionMat = cv::Mat::zeros(3, 3, CV_32F);
    cv::Matx33d projection = projectionMat;
    std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    projection(0, 0) = focal_length;
    projection(1, 1) = focal_length;
    projection(0, 2) = center.x;
    projection(1, 2) = center.y;
    projection(2, 2) = 1;

    std::vector<cv::Point3d> head_points;

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
//    head_points.push_back(P3D_SUBNASALE);
    head_points.push_back(P3D_STOMMION);
    std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    std::vector<type_point> detected_points;

    detected_points.push_back(coordsOf(shape, SELLION));
    detected_points.push_back(coordsOf(shape, RIGHT_EYE));
    detected_points.push_back(coordsOf(shape, LEFT_EYE));
    detected_points.push_back(coordsOf(shape, RIGHT_SIDE));
    detected_points.push_back(coordsOf(shape, LEFT_SIDE));
    detected_points.push_back(coordsOf(shape, MENTON));
    detected_points.push_back(coordsOf(shape, NOSE));
//    detected_points.push_back(coordsOf(shape, SUBNASALE));

    auto stomion = (coordsOf(shape, MOUTH_CENTER_TOP) + coordsOf(shape, MOUTH_CENTER_BOTTOM)) * 0.5;
    detected_points.push_back(stomion);
std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    cv::Mat rotation_vector, translation_vector;

    // Find the 3D pose of our head
    cv::solvePnP(head_points, detected_points,
             projection, cv::noArray(),
             rotation_vector, translation_vector, false,
#ifdef OPENCV3
             cv::SOLVEPNP_ITERATIVE);
#else
             cv::ITERATIVE);
#endif
std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    cv::Matx33d rotation;
    cv::Rodrigues(rotation_vector, rotation);
std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    head_pose pose = {
        rotation(0, 0), rotation(0, 1), rotation(0, 2), translation_vector.at<double>(0) / 1000,
        rotation(1, 0), rotation(1, 1), rotation(1, 2), translation_vector.at<double>(1) / 1000,
        rotation(2, 0), rotation(2, 1), rotation(2, 2), translation_vector.at<double>(2) / 1000,
        0, 0, 0, 1};

#ifdef HEAD_POSE_ESTIMATION_DEBUG

    std::vector<type_point> reprojected_points;

    cv::projectPoints(head_points, rotation_vector, translation_vector, projection, cv::noArray(), reprojected_points);

    for (auto point : reprojected_points)
    {
        cv::circle(_debug, point, 2, cv::Scalar(0, 255, 255), 2);
    }
std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    std::vector<cv::Point3d> axes;
    axes.push_back(cv::Point3d(0, 0, 0));
    axes.push_back(cv::Point3d(50, 0, 0));
    axes.push_back(cv::Point3d(0, 50, 0));
    axes.push_back(cv::Point3d(0, 0, 50));
    std::vector<type_point> projected_axes;

    cv::projectPoints(axes, rotation_vector, translation_vector, projection, cv::noArray(), projected_axes);
std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    cv::line(_debug, projected_axes[0], projected_axes[3], cv::Scalar(255, 0, 0), 2, CV_AA);
    cv::line(_debug, projected_axes[0], projected_axes[2], cv::Scalar(0, 255, 0), 2, CV_AA);
    cv::line(_debug, projected_axes[0], projected_axes[1], cv::Scalar(0, 0, 255), 2, CV_AA);

    // putText(_debug, "(" + to_string(int(pose(0, 3) * 100)) + "cm, " + to_string(int(pose(1, 3) * 100)) + "cm, " + to_string(int(pose(2, 3) * 100)) + "cm)", coordsOf(shape, SELLION), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
#endif
std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    cv::Matx34d projection_matrix = {
        rotation(0, 0), rotation(0, 1), rotation(0, 2), 0,
        rotation(1, 0), rotation(1, 1), rotation(1, 2), 0,
        rotation(2, 0), rotation(2, 1), rotation(2, 2), 0};

    cv::Vec3d eulerAngles;

    std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    decomposeProjectionMatrix(projection_matrix, projection, rotation, translation_vector,
                              cv::noArray(), cv::noArray(), cv::noArray(), eulerAngles);
    yaw = eulerAngles[1];
    pitch = eulerAngles[0];
    roll = eulerAngles[2];

    cv::Mat rotation2;
    cv::Rodrigues(rotation_vector, rotation2);
    cv::Vec3d eulerAngles2 = rotationMatrixToEulerAngles(rotation2);

    roll2 = eulerAngles2[0];
    pitch2 = eulerAngles2[1];
    yaw2 = eulerAngles2[2];

    std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
    return pose;
}

face_cropper::face_cropper()
{
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
}

template <typename T>
void face_cropper::detect(dlib::cv_image<T> &image)
{
    double focal_length = image.nc();
    cv::Point2d center = cv::Point2d(image.nc() / 2., image.nr() / 2);

    faces.clear();
    std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;
//#pragma omp critical
    {
        faces = detector(image);
    }
    std::cerr << omp_get_thread_num() << "\t" << __LINE__ << std::endl;

    shapes.clear();
    metrics.clear();
    for (std::vector<dlib::rectangle>::iterator it = faces.begin(); it != faces.end(); ++it)
    {
        shapes.push_back(predictor(image, *it));
        metrics.push_back(face_metrics(focal_length, center, shapes.back()));
    }
}
template void face_cropper::detect<dlib::bgr_pixel>(dlib::cv_image<dlib::bgr_pixel>&);

int face_cropper::get_num_faces()
{
    return faces.size();
}

// https://qiita.com/vs4sh/items/93d65468a992af5b8f92
void face_cropper::crop_rotatedrect(cv::Mat &i_img, cv::RotatedRect &rect, cv::Mat &o_img)
{
    cv::Mat rotation_matrix, rotated = i_img;

    angle = rect.angle;
    cv::Size rect_size = rect.size;
    if (rect.angle < -45.)
    {
        angle += 90.0;
        std::swap(rect_size.width, rect_size.height);
    }
    // std::cerr << angle << std::endl;

    // 回転矩形の角度から回転行列を計算
    rotation_matrix = cv::getRotationMatrix2D(rect.center, angle, 1.0);
    // 元画像を回転
    cv::warpAffine(i_img, rotated, rotation_matrix, rotated.size(), cv::INTER_CUBIC);
    // 回転した画像から矩形領域を切り出す
    // o_img = cv::Mat(i_img.size(), i_img.type());
    o_img = i_img;
    cv::getRectSubPix(rotated, rect_size, rect.center, o_img);
}

void face_cropper::crop_nth(cv::Mat &i_img, int n, cv::Mat &o_img)
{
    std::vector<type_point> crop_rect = metrics[n].get_crop_rect();

    cv::RotatedRect rect;
#if HAVE_ROTATEDRECT_3PT
    rect = cv::RotatedRect(crop_rect[0], crop_rect[1], crop_rect[2]);
#else
    rect = RotatedRect_pt(crop_rect[0], crop_rect[1], crop_rect[2]);
#endif
    crop_rotatedrect(i_img, rect, o_img);
}

void face_cropper::dump_metric(int n, std::ostream &os)
{
    metrics[n].dump_metric(os);

    std::vector<type_point> crop_rect = metrics[n].get_crop_rect();
    cv::RotatedRect rect;
#if HAVE_ROTATEDRECT_3PT
    rect = RotatedRect(crop_rect[0], crop_rect[1], crop_rect[2]);
#else
    rect = RotatedRect_pt(crop_rect[0], crop_rect[1], crop_rect[2]);
#endif
    angle = rect.angle;
    if (angle < -45.)
    {
        angle += 90.0;
    }
    os << "\t" << angle;
}

bool face_cropper::detect_and_output(std::string &read_img_name, std::string &write_img_name, std::ostringstream &oss)
{
    bool read_recognize_error = false;
    std::string filename;
    cv::Mat img_color = cv::imread(read_img_name), o_img;
    if (img_color.data == NULL)
    {
        read_recognize_error = true;
        oss << "Not an image or not supported";
    }
    else
    {
        dlib::cv_image<dlib::bgr_pixel> cimg(img_color);
        detect(cimg);

        if (0 == get_num_faces())
        {
            read_recognize_error = true;
            oss << "no face is detected in: " << read_img_name << std::endl;
        }
        for (int i = 0; i < get_num_faces(); ++i)
        {
            // std::cerr << __LINE__ << std::endl;
            crop_nth(img_color, i, o_img);
            filename = get_nth_img_name(write_img_name, i);
            cv::imwrite(filename.c_str(), o_img);
            // std::cerr << "write: " << filename << std::endl;
            oss << filename << "\t";
            dump_metric(i, oss);
        }
    }
    return read_recognize_error;
}
