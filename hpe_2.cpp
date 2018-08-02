#include <vector>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <iterator>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace cv;

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

cv::Point2d toCv(const dlib::point &p)
{
    return cv::Point2d(p.x(), p.y());
}

cv::Point2d coordsOf(dlib::full_object_detection &shape, FACIAL_FEATURE feature)
{
    return toCv(shape.part(feature));
}

int main(int argc, char **argv)
{

    // Read input image
    cv::Mat im;
    if (argc < 2)
    {
        im = cv::imread("headPose.jpg");
    }
    else
    {
        im = cv::imread(argv[1]);
    }

    // 2D image points. If you change the image, you need to change vector
    std::vector<cv::Point2d> image_points;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor predictor;
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    std::vector<dlib::rectangle> faces;
    std::vector<dlib::full_object_detection> shapes;

    dlib::cv_image<dlib::bgr_pixel> cimg(im);
    faces = detector(cimg);

    for (std::vector<dlib::rectangle>::iterator it = faces.begin(); it != faces.end(); ++it)
    {
        shapes.push_back(predictor(cimg, *it));
    }

    image_points.push_back(coordsOf(shapes[0], NOSE));              // Nose tip
    image_points.push_back(coordsOf(shapes[0], MENTON));            // Chin
    image_points.push_back(coordsOf(shapes[0], LEFT_EYE));          // Left eye left corner
    image_points.push_back(coordsOf(shapes[0], RIGHT_EYE));         // Right eye right corner
    image_points.push_back(coordsOf(shapes[0], MOUTH_LEFT));        // Left Mouth corner
    image_points.push_back(coordsOf(shapes[0], MOUTH_RIGHT));       // Right mouth corner

    /*
    image_points.push_back(cv::Point2d(359, 391)); // Nose tip
    image_points.push_back(cv::Point2d(399, 561)); // Chin
    image_points.push_back(cv::Point2d(337, 297)); // Left eye left corner
    image_points.push_back(cv::Point2d(513, 301)); // Right eye right corner
    image_points.push_back(cv::Point2d(345, 465)); // Left Mouth corner
    image_points.push_back(cv::Point2d(453, 469)); // Right mouth corner
*/
    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));          // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));     // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));  // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));   // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));  // Right mouth corner

    // Camera internals
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = cv::Point2d(im.cols / 2, im.rows / 2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

    cout << "Camera Matrix " << endl
         << camera_matrix << endl;
    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;

    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    // Project a 3D point (0, 0, 1000.0) onto the image plane.
    // We use this to draw a line sticking out of the nose

    vector<Point3d> nose_end_point3D;
    vector<Point2d> nose_end_point2D;
    nose_end_point3D.push_back(Point3d(0, 0, 1000.0));

    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

    for (int i = 0; i < image_points.size(); i++)
    {
        circle(im, image_points[i], 3, Scalar(0, 0, 255), -1);
    }

    cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

    cout << "Rotation Vector " << endl
         << rotation_vector << endl;
    cout << "Translation Vector" << endl
         << translation_vector << endl;

    cout << nose_end_point2D << endl;

    cv::Matx33d rotation;
    cv::Rodrigues(rotation_vector, rotation);
    cv::Matx34d projection_matrix = {
        rotation(0, 0), rotation(0, 1), rotation(0, 2), 0,
        rotation(1, 0), rotation(1, 1), rotation(1, 2), 0,
        rotation(2, 0), rotation(2, 1), rotation(2, 2), 0};

    cv::Vec3d eulerAngles;
    std::cerr << __LINE__ << std::endl;
    decomposeProjectionMatrix(projection_matrix, camera_matrix, rotation, translation_vector,
                              cv::noArray(), cv::noArray(), cv::noArray(), eulerAngles);
    double yaw, pitch, roll;
    yaw = eulerAngles[1];
    pitch = eulerAngles[0];
    roll = eulerAngles[2];

    cout << "roll, pitch, yaw\t" << roll << "\t" << pitch << "\t" << yaw << endl;

    // Display image.
    // cv::imshow("Output", im);
    cv::imwrite("hpe_result.jpg", im);
    // cv::waitKey(0);
}
