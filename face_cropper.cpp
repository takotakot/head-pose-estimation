#include <vector>
#include <cstdio>
#include <sstream>
#include <iterator>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

// using namespace dlib;
// using namespace std;

typedef cv::Point2f type_point;

struct face_metrics
{
    const int chin_index = 8;
    cv::Mat uc, c0, c1, ue, mo2;
    const double L4 = 231.9;
    const double L16 = 121.1;
    const double L8 = 160.8;
    double l16;
    double l4, l4mod, l8;

    face_metrics(dlib::full_object_detection &shape)
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
        l4 = l16 * l4 / L16;
        l4mod = l4 * 1.1;
        l8 = l16 * L8 / L16;
    }

    cv::Mat get_points_index(int index[], const int &n_index, dlib::full_object_detection &shape)
    {
        cv::Mat point_matrix(cv::Size(2, n_index), CV_64FC1);
        for (int i = 0; i < n_index; ++i)
        {
            point_matrix.at<double>(i, 0) = shape.part(index[i]).x();
            point_matrix.at<double>(i, 1) = shape.part(index[i]).y();
        }
   
        return point_matrix;
    }

    cv::Mat get_center_points(dlib::full_object_detection &shape)
    {
        // center 10
        int center_points_index[] = {27, 28, 29, 30, 33, 51, 62, 66, 57, 8};
        const int n_center_points = sizeof(center_points_index) / sizeof(center_points_index[0]);

        return get_points_index(center_points_index, n_center_points, shape);
    }

    cv::Mat get_eye_points(dlib::full_object_detection &shape)
    {
        // eye line 13
        int eye_points_index[] = {36, 37, 38, 39, 40, 41, 27, 42, 43, 44, 45, 46, 47};
        const int n_eye_points = sizeof(eye_points_index) / sizeof(eye_points_index[0]);

        return get_points_index(eye_points_index, n_eye_points, shape);
    }

    cv::Mat get_chin(dlib::full_object_detection &shape)
    {
        cv::Mat point_matrix(cv::Size(2, 1), CV_64FC1);
        point_matrix.at<double>(1, 0) = shape.part(chin_index).x();
        point_matrix.at<double>(1, 1) = shape.part(chin_index).y();

        return point_matrix;
    }

    std::vector<type_point> get_face_rect() {
        cv::Mat tmp;
        std::vector<type_point> rect;
        
        tmp = mo2 - (l4mod - l16) * c0 - (l8/2) * c1;
        rect.push_back(type_point(tmp));

        tmp += l4mod * c0;
        rect.push_back(type_point(tmp));

        tmp += l8 * c1;
        rect.push_back(type_point(tmp));

        tmp += -l4mod * c1;
        rect.push_back(type_point(tmp));

        return rect;
    }

    std::vector<type_point> get_crop_rect() {
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
};

class face_cropper
{
    dlib::frontal_face_detector detector;
    dlib::shape_predictor predictor;
    std::vector<dlib::rectangle> faces;
    std::vector<dlib::full_object_detection> shapes;
    const int n_landmarks = 68;

  public:
    face_cropper()
    {
        detector = dlib::get_frontal_face_detector();
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    }

    template <typename T>
    void detect(dlib::cv_image<T> &image)
    {
        faces.clear();
        faces = detector(image);

        shapes.clear();
        for (std::vector<dlib::rectangle>::iterator it = faces.begin(); it != faces.end(); ++it)
        {
            shapes.push_back(predictor(image, *it));
        }
    }

    int get_num_faces()
    {
        return faces.size();
    }

    void crop_nth(cv::Mat &i_img, int n, cv::Mat &o_img)
    {
        face_metrics metrics(shapes[n]);
    }
};

int main(int argc, char *argv[])
{

    // thread_local
    face_cropper cropper;
    char buffer[5];

    //text on screen
    std::ostringstream outtext;
    std::cerr << __LINE__ << std::endl;

    //main loop
    do
    {
        // Grab a frame
        cv::Mat temp, o_img;
        cv::Mat cropped;
        std::string filename;
        // cap >> temp;
        if (argc < 2)
        {
            temp = cv::imread("IMG_4899.jpg");
        }
        else
        {
            temp = cv::imread(argv[1]);
        }
        dlib::cv_image<dlib::bgr_pixel> cimg(temp);

        cropper.detect(cimg);

        std::cerr << __LINE__ << std::endl;

        for (int i = 0; i < cropper.get_num_faces(); ++i)
        {
            std::cerr << __LINE__ << std::endl;
            cropper.crop_nth(temp, i, o_img);
            std::snprintf(buffer, sizeof(buffer), "_%d", i);
            filename = "result";
            filename += buffer;
            filename += ".jpg";
            cv::imwrite(filename.c_str(), o_img);
        }
        std::cerr << __LINE__ << std::endl;
    } while (false);

    return 0;
}
