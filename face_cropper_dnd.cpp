#include "face_cropper.h"
#include "read_srcs.h"
#include <ctime>

struct face_detect_result
{
    // int w, h;
    int rx, ry;
    int rw, rh;
    int detected; // 0|1
};

std::string in_dir_name = "to_recognize/";
std::string out_dir_name = "dlib_results/";
std::string file_list_name = "to_recognize_list_all.txt";
std::string error_file_name = "face_recognition_error.txt";
// std::string dat_file_name = "recog.dat";
std::string result_file_name = "result_metric.txt";

int main(int argc, char *argv[])
{
    time_t timer;
    timer = time(NULL);
    struct tm *timeinfo;
    timeinfo = localtime(&timer);
    char sbuf[256];
    std::strftime(sbuf, 256, "_%F_%H%M", timeinfo);
    struct src_set sst;

    std::string dir_str = "c:\\cygwin64\\";
    if (argc < 2)
    {
        // nop
    }
    else
    {
        for (int i = 1; i < argc; ++i)
        {
            dir_str = argv[i];
            sst.add(dir_str);
        }
    }

    // std::cerr << "begin" << std::endl;
    std::string buf;
    int w, h;
    face_detect_result prev_result;
    // read file list
    std::vector<std::string> file_list;
    std::vector<int> ws, hs;
    std::vector<face_detect_result> results;
    std::ifstream ifs;
    std::ofstream ofs;
    ifs.open(file_list_name.c_str());
    ofs.open(result_file_name.c_str());
    while (!ifs.eof())
    {
        // std::getline(ifs, buf);
        ifs >> buf >> w >> h;
        ifs >> prev_result.rx >> prev_result.ry >> prev_result.rw >> prev_result.rh >> prev_result.detected;
        // if(buf != "") {
        if (ifs)
        {
            file_list.push_back(buf);
            ws.push_back(w);
            hs.push_back(h);
            results.push_back(prev_result);
        }
        // std::cerr << buf << std::endl;
    }
    ifs.close();

    std::ofstream ofs_error;
    ofs_error.open(error_file_name.c_str());

    /*
	*/
    // #pragma omp parallel for private(read_img_name, write_img_name, img_color, recog_ok, cropper_obj)
    // private(read_img_name, write_img_name, img_color, recog_ok)
    // #pragma omp parallel
    {
        // cropper cropper_obj;
        face_cropper cropper;
        char buffer[5];
        std::stringstream obuf;
        cv::Mat img_color, o_img;
        
        std::string filename, read_img_name, write_img_name;
        bool recog_ok;
        std::ostringstream oss;

        //#pragma omp for
        for (int i = 0; i < file_list.size(); ++i)
        {
            oss.str("");
            // http://stackoverflow.com/questions/15033827/multiple-threads-writing-to-stdcout-or-stdcerr
            std::cerr << file_list[i] << std::endl;

            // name
            read_img_name = in_dir_name + file_list[i];
            write_img_name = out_dir_name + file_list[i];

            recog_ok = false;

            // read
            img_color = cv::imread(read_img_name);
            dlib::cv_image<dlib::bgr_pixel> cimg(img_color);

            cropper.detect(cimg);

            for (int i = 0; i < cropper.get_num_faces(); ++i)
            {
                // std::cerr << __LINE__ << std::endl;
                cropper.crop_nth(img_color, i, o_img);
                if (i > 0)
                {
                    std::snprintf(buffer, sizeof(buffer), "_%d", i);
                    filename = write_img_name;
                    filename += buffer;
                    filename += ".jpg";
                    cv::imwrite(filename.c_str(), o_img);
                }
                else
                {
                    filename = write_img_name;
                    cv::imwrite(write_img_name.c_str(), o_img);
                }
                // std::cerr << "write: " << filename << std::endl;
                oss << filename << "\t";
                cropper.dump_metric(i, oss);
                std::cerr << oss.str() << std::endl;
                ofs << oss.str() << std::endl;
            }
        }
        // #pragma omp critical
        {
            //            ofs << obuf.str();
        }
    }

    return 0;
}
