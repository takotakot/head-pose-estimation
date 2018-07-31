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

class face_cropper {
    dlib::frontal_face_detector detector;
    dlib::shape_predictor predictor;
    std::vector<dlib::rectangle> faces;
    std::vector<dlib::full_object_detection> shapes;

  public:
    face_cropper()
    {
        detector = dlib::get_frontal_face_detector();
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    }

    template<typename T> void detect(dlib::cv_image<T> &image)
    {
        faces.clear();
        faces = detector(image);

        shapes.clear();
        for (std::vector<dlib::rectangle>::iterator it = faces.begin(); it != faces.end(); ++it)
        {
            shapes.push_back(predictor(image, *it));
        }
    }
	
	int get_num_faces() {
		return faces.size();
	}
	
	void crop_nth(cv::Mat &i_img, int n, cv::Mat &o_img) {
        o_img = i_img;
    }
};

int main(int argc, char* argv[])
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
        }else{
        	temp = cv::imread(argv[1]);
        }
        dlib::cv_image<dlib::bgr_pixel> cimg(temp);

        cropper.detect(cimg);

    	std::cerr << __LINE__ << std::endl;

    	for(int i = 0; i < cropper.get_num_faces(); ++i) {
            std::cerr << __LINE__ << std::endl;
            cropper.crop_nth(temp, i, o_img);
            std::snprintf(buffer, sizeof(buffer), "_%d", i);
            filename = "result";
            filename += buffer;
            filename += ".jpg";
            cv::imwrite(filename.c_str(), o_img);
        }
        std::cerr << __LINE__ << std::endl;
    }while (false);


    return 0;
}
