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

// #define FPOINT(x, y, z) ((y), -(z), -(x))
#define FPOINT(x, y, z) ((y), -(z), (x))

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

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
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
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{

    assert(isRotationMatrix(R));

    float sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
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
    return cv::Vec3f(x, y, z);
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

/*
    image_points.push_back(coordsOf(shapes[0], NOSE));              // Nose tip
    image_points.push_back(coordsOf(shapes[0], MENTON));            // Chin
    image_points.push_back(coordsOf(shapes[0], LEFT_EYE));          // Left eye left corner
    image_points.push_back(coordsOf(shapes[0], RIGHT_EYE));         // Right eye right corner
    image_points.push_back(coordsOf(shapes[0], MOUTH_LEFT));        // Left Mouth corner
    image_points.push_back(coordsOf(shapes[0], MOUTH_RIGHT));       // Right mouth corner
*/

    auto stomion = (coordsOf(shapes[0], MOUTH_CENTER_TOP) + coordsOf(shapes[0], MOUTH_CENTER_BOTTOM)) * 0.5;
    image_points.push_back(coordsOf(shapes[0], SELLION));     // Sellion
    image_points.push_back(coordsOf(shapes[0], MENTON));      // Chin
    image_points.push_back(coordsOf(shapes[0], LEFT_EYE));    // Left eye left corner
    image_points.push_back(coordsOf(shapes[0], RIGHT_EYE));   // Right eye right corner
    image_points.push_back(coordsOf(shapes[0], LEFT_SIDE));   // Left Mouth corner
    image_points.push_back(coordsOf(shapes[0], RIGHT_SIDE));  // Right mouth corner
    image_points.push_back(coordsOf(shapes[0], NOSE));
    image_points.push_back(coordsOf(shapes[0], SUBNASALE));
    image_points.push_back(coordsOf(shapes[0], MOUTH_LEFT));  // Left Mouth corner
    image_points.push_back(coordsOf(shapes[0], MOUTH_RIGHT)); // Right mouth corner
    image_points.push_back(stomion);

    // 3D model points.
    std::vector<cv::Point3d> model_points;
/*
    const static cv::Point3d P3D_SELLION(0., 0., 0.);
    const static cv::Point3d P3D_LEFT_EYE(-20., 65.5, -5.);
    const static cv::Point3d P3D_RIGHT_EYE(-20., -65.5, -5.);
    const static cv::Point3d P3D_LEFT_EAR(-100., 77.5, -6.);
    const static cv::Point3d P3D_RIGHT_EAR(-100., -77.5, -6.);
    const static cv::Point3d P3D_NOSE(21.0, 0., -48.0);
    const static cv::Point3d P3D_STOMION(10.0, 0., -75.0);
    const static cv::Point3d P3D_MENTON(0., 0., -133.0);
    const static cv::Point3d P3D_MOUTH_LEFT(10.0, 27.9, -75.0);
    const static cv::Point3d P3D_MOUTH_RIGHT(10.0, -27.9, -75.0);
    */
    const static cv::Point3f P3D_SELLION FPOINT(0., 0., 0.);
    const static cv::Point3f P3D_LEFT_EYE FPOINT(-20., 45.55, -5.);
    const static cv::Point3f P3D_RIGHT_EYE FPOINT(-20., -45.55, -5.);
    const static cv::Point3f P3D_LEFT_EAR FPOINT(-100., 74.25, -6.);
    const static cv::Point3f P3D_RIGHT_EAR FPOINT(-100., -74.25, -6.);
    const static cv::Point3f P3D_NOSE FPOINT(21.0, 0., -48.0);
    const static cv::Point3f P3D_SUBNASALE FPOINT(0., 0., -48.0);
    const static cv::Point3f P3D_STOMION FPOINT(10.0, 0., -75.0);
    const static cv::Point3f P3D_MENTON FPOINT(-32.14, 0., -116.76);
    const static cv::Point3f P3D_MOUTH_LEFT FPOINT(-32.14, 27.9, -116.76);
    const static cv::Point3f P3D_MOUTH_RIGHT FPOINT(-32.14, -27.9, -116.76);

    const static cv::Point3f P3D_HEAD_TOP FPOINT(-100., 0, 115.14);
    const static cv::Point3f P3D_HEAD_BACK FPOINT(-70.7, 0, 115.14);

    model_points.push_back(P3D_SELLION);
    model_points.push_back(P3D_MENTON);
    model_points.push_back(P3D_LEFT_EYE);
    model_points.push_back(P3D_RIGHT_EYE);
    model_points.push_back(P3D_LEFT_EAR);
    model_points.push_back(P3D_RIGHT_EAR);
    model_points.push_back(P3D_NOSE);
    model_points.push_back(P3D_SUBNASALE);
    model_points.push_back(P3D_MOUTH_LEFT);
    model_points.push_back(P3D_MOUTH_RIGHT);
    model_points.push_back(P3D_STOMION);

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
    // nose_end_point3D.push_back(Point3d(0, 0, 1000.0));
    nose_end_point3D.push_back(Point3d FPOINT(100, 0, 0));
    nose_end_point3D.push_back(Point3d FPOINT(0, 100, 0));
    nose_end_point3D.push_back(Point3d FPOINT(0, 0, -100));
    nose_end_point3D.push_back(P3D_HEAD_TOP);
    nose_end_point3D.push_back(P3D_HEAD_BACK);

    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

    for (int i = 0; i < image_points.size(); i++)
    {
        circle(im, image_points[i], 3, Scalar(0, 0, 255), -1);
    }

    cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);
    cv::line(im, coordsOf(shapes[0], LEFT_SIDE), nose_end_point2D[1], cv::Scalar(255, 0, 0), 2);
    cv::line(im, coordsOf(shapes[0], MENTON), nose_end_point2D[2], cv::Scalar(255, 0, 0), 2);
    cv::line(im, image_points[0], nose_end_point2D[3], cv::Scalar(255, 0, 0), 2);
    cv::line(im, nose_end_point2D[3], nose_end_point2D[4], cv::Scalar(0, 255, 0), 2);

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

    cv::Mat rotation2;
    cv::Rodrigues(rotation_vector, rotation2);
    cv::Vec3d eulerAngles2 = rotationMatrixToEulerAngles(rotation2);

    cout << "Rotation Matrix " << endl
         << rotation << endl;

    cv::Vec3d eulerAngles;
    decomposeProjectionMatrix(projection_matrix, camera_matrix, rotation, translation_vector,
                              cv::noArray(), cv::noArray(), cv::noArray(), eulerAngles);
    double yaw, pitch, roll;

    cout << "Camera Matrix " << endl
         << camera_matrix << endl;
    cout << "Rotation Matrix " << endl
         << rotation << endl;
    cout << "trans " << endl
         << translation_vector << endl;
    cout << "euler " << endl
         << eulerAngles << endl;


    // https://stackoverflow.com/questions/27508242/roll-pitch-and-yaw-from-rotation-matrix-with-eigen-library
    // https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    yaw = eulerAngles[1];
    pitch = eulerAngles[0];
    roll = eulerAngles[2];

    cout << "roll, pitch, yaw\t" << roll << "\t" << pitch << "\t" << yaw << endl;
    cout << "roll, pitch, yaw\t" << eulerAngles2[0] << "\t" << eulerAngles2[1] << "\t" << eulerAngles2[2] << endl;

    // Display image.
    // cv::imshow("Output", im);
    cv::imwrite("hpe_result.jpg", im);
    // cv::waitKey(0);
}
