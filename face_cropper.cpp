#include "face_cropper.h"

#define HAVE_ROTATEDRECT_3PT 0

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

face_metrics::face_metrics(double cols, double rows, dlib::full_object_detection &shape): focal_length(focal_length), center(center)
{
    cv::Mat center_points = get_center_points(shape);
    cv::Mat eye_points = get_eye_points(shape);

    cv::reduce(center_points, uc, 0, CV_REDUCE_AVG);
    cv::PCA pca_center_points(center_points, cv::Mat(), CV_PCA_DATA_AS_ROW, 2);
    c0 = pca_center_points.eigenvectors.row(0);
    c1 = pca_center_points.eigenvectors.row(1);
    cv::reduce(eye_points, ue, 0, CV_REDUCE_AVG);

    mo2 = (ue - uc) * c0.t() * c0 + uc;

    l16 = c0.dot(get_chin(shape) - mo2);
    double dx = shape.part(16).x() - shape.part(0).x(), dy = shape.part(16).y() - shape.part(0).y();
    l11 = std::sqrt(dx * dx + dy * dy);
    l4 = l16 * length_4 / length_16;
    l8 = l16 * length_8 / length_16;
    // heuristic: if l16 is too small compared to l11, set l4mod longer
    // TODO: refine
    l4mod = l4 * 1.2;
    std::cerr << l16 << std::endl;
    // std::cerr << l4 << std::endl;

    pose = face_metrics::calc_pose(shape);
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

/*
        Refactorization is needed:
        http://tessy.org/wiki/index.php?%B9%D4%CE%F3%A4%CE%BE%E8%BB%BB%A1%A4%C6%E2%C0%D1%A1%A4%B3%B0%C0%D1
        https://qiita.com/fukushima1981/items/d283b3af3e21d94550c4
        https://qiita.com/ChaoticActivity/items/68f10d7452680fa1d52d
    */
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

std::vector<type_point> face_metrics::get_crop_rect()
{
    const double m1 = 40, m2 = 30, m3 = 3.5, m4 = 30;
    cv::Mat tmp;
    std::vector<type_point> rect;

    tmp = mo2 - (l4mod + l4mod * m3 / m4 - l16) * c0 - (l4mod * m2 / m4 / 2) * c1;
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
    os << l16 << "\t" << l11 << "\t" << l11 / l16;
}

head_pose face_metrics::calc_pose(dlib::full_object_detection &shape) {
    return head_pose();
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
    faces = detector(image);

    shapes.clear();
    metrics.clear();
    for (std::vector<dlib::rectangle>::iterator it = faces.begin(); it != faces.end(); ++it)
    {
        shapes.push_back(predictor(image, *it));
        metrics.push_back(shapes.back());
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
    cv::Mat rotation_matrix, rotated;

    float angle = rect.angle;
    cv::Size rect_size = rect.size;
    if (rect.angle < -45.)
    {
        angle += 90.0;
        std::swap(rect_size.width, rect_size.height);
    }
    std::cerr << angle << std::endl;

    // 回転矩形の角度から回転行列を計算
    rotation_matrix = cv::getRotationMatrix2D(rect.center, angle, 1.0);
    // 元画像を回転
    cv::warpAffine(i_img, rotated, rotation_matrix, i_img.size(), cv::INTER_CUBIC);
    // 回転した画像から矩形領域を切り出す
    cv::getRectSubPix(rotated, rect_size, rect.center, o_img);
}

void face_cropper::crop_nth(cv::Mat &i_img, int n, cv::Mat &o_img)
{
    std::vector<type_point> crop_rect = metrics[n].get_crop_rect();

    cv::RotatedRect rect;
#if HAVE_ROTATEDRECT_3PT
    rect = RotatedRect(crop_rect[0], crop_rect[1], crop_rect[2]);
#else
    rect = RotatedRect_pt(crop_rect[0], crop_rect[1], crop_rect[2]);
#endif
    crop_rotatedrect(i_img, rect, o_img);
}

void face_cropper::dump_metric(int n, std::ostream &os)
{
    metrics[n].dump_metric(os);
}
