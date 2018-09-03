#include "face_cropper.h"

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

            cropper.dump_metric(i, std::cerr);
            std::cerr << std::endl;
        }
        std::cerr << __LINE__ << std::endl;
    } while (false);

    return 0;
}
