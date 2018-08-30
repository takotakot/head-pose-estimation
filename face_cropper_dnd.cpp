#include "face_cropper.h"
#include "read_srcs.h"
#include "functions.h"
#include <ctime>
#include<sys/stat.h>

struct face_detect_result
{
    // int w, h;
    int rx, ry;
    int rw, rh;
    int detected; // 0|1
};

int main(int argc, char *argv[])
{
    time_t timer;
    timer = time(NULL);
    struct tm *timeinfo;
    timeinfo = localtime(&timer);
    char sbuf[256];
    std::strftime(sbuf, 256, "_%F_%H%M", timeinfo);
    std::string date_suffix(sbuf);
    struct src_set sst;

    std::string dir_str = "c:\\cygwin64\\";
    if (argc < 2)
    {
        // nop
        return 0;
    }
    else
    {
        for (int i = 1; i < argc; ++i)
        {
            dir_str = argv[i];
            sst.add(dir_str);
        }
    }

    sst.dump(std::cerr);
    create_all(date_suffix, sst);

    std::string error_file_name = "face_recognition_error.txt";
    std::string result_file_name = "result_metric.txt";

    std::string buf;
    int w, h;
    face_detect_result prev_result;
    // read file list
    std::vector<face_detect_result> results;
    std::ofstream ofs;
    ofs.open(result_file_name.c_str());
    //        file_list.push_back(buf);
    std::ofstream ofs_error;
    ofs_error.open(error_file_name.c_str());

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
        std::ostringstream oss;

        //#pragma omp for
        for (int i = 0; i < sst.srcs.front().dir.filelist.size(); ++i)
        {
            oss.str("");
            // http://stackoverflow.com/questions/15033827/multiple-threads-writing-to-stdcout-or-stdcerr
            std::cerr << sst.srcs.front().dir.filelist[i] << std::endl;

            // name
//            read_img_name = in_dir_name + file_list[i];
//            write_img_name = out_dir_name + file_list[i];

if(0){
            // read
            read_img_name = sst.srcs.front().dir.filelist[i];
            img_color = cv::imread(read_img_name);
            dlib::cv_image<dlib::bgr_pixel> cimg(img_color);

            cropper.detect(cimg);

            if ( 0 == cropper.get_num_faces()) {
                std::cerr << "no face is detected in: " << read_img_name << std::endl;
            }
            for (int i = 0; i < cropper.get_num_faces(); ++i)
            {
                // std::cerr << __LINE__ << std::endl;
                cropper.crop_nth(img_color, i, o_img);
                // filename = get_write_img_name(write_img_name, i);
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
        }
        // #pragma omp critical
        {
            //            ofs << obuf.str();
        }
    }

    return 0;
}
