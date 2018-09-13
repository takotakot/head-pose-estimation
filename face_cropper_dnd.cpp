#include "face_cropper.h"
#include "read_srcs.h"
#include "functions.h"

#include <omp.h>
#include <sys/stat.h>

#ifdef __CYGWIN__
#include <sys/cygwin.h>
#endif

int main(int argc, char *argv[])
{
    std::string date_suffix(get_date_suffix());
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
            std::cerr << dir_str;
#ifdef __CYGWIN__
            ssize_t size;
            char *posix;
            size = cygwin_conv_path(CCP_WIN_A_TO_POSIX | CCP_PROC_CYGDRIVE, dir_str.c_str(), NULL, 0);
            if (size < 0)
            {
                perror("cygwin_conv_path");
            }
            else
            {
                posix = (char *)malloc(size);
                if (cygwin_conv_path(CCP_WIN_A_TO_POSIX | CCP_PROC_CYGDRIVE, dir_str.c_str(), posix, size))
                    perror("cygwin_conv_path");
            }
            dir_str = posix;
            std::cerr << "\t" << dir_str << std::endl;
            free(posix);
#endif
            sst.add(dir_str);
        }
    }

    sst.dump(std::cerr);
    create_all(date_suffix, sst);

    std::string error_file_name = "face_recognition_error" + date_suffix + ".txt";
    std::string result_file_name = "result_metric" + date_suffix + ".txt";

    std::string buf;
    // read file list
    std::ofstream ofs;
    ofs.open(result_file_name.c_str());
    //        file_list.push_back(buf);
    std::ofstream ofs_error;
    ofs_error.open(error_file_name.c_str());

// #pragma omp parallel for private(read_img_name, write_img_name, img_color, recog_ok, cropper_obj)
// private(read_img_name, write_img_name, img_color, recog_ok)
#pragma omp parallel
    {
        // cropper cropper_obj;
        face_cropper cropper;
        // std::stringstream obuf;
        cv::Mat img_color, o_img;

        std::string filename, read_img_name, write_img_name, dest_dirname;
        std::ostringstream oss;
        int thread_num = omp_get_thread_num();
        bool read_recognize_error = false;

        // #pragma omp for
        for (auto src : sst.srcs)
        {
            if (src.type == 1)
            {
#pragma omp single
                {
                    // TODO: treat single file
                    oss.str("");
                    read_img_name = src.filename;
                    read_recognize_error = false;
#pragma omp critical
                    {
                        std::cerr << "th" << thread_num << ": " << read_img_name << std::endl;
                    }
                    write_img_name = append_date_suffix(read_img_name, date_suffix);
                    // std::cerr << write_img_name << std::endl;

                    // code duplicated
                    img_color = cv::imread(read_img_name);
                    if (img_color.data == NULL)
                    {
                        read_recognize_error = true;
                        oss << "Not an image or not supported";
                    }
                    else
                    {
                        dlib::cv_image<dlib::bgr_pixel> cimg(img_color);

                        cropper.detect(cimg);

                        if (0 == cropper.get_num_faces())
                        {
                            read_recognize_error = true;
                            oss << "no face is detected in: " << read_img_name << std::endl;
                        }
                        for (int i = 0; i < cropper.get_num_faces(); ++i)
                        {
                            // std::cerr << __LINE__ << std::endl;
                            cropper.crop_nth(img_color, i, o_img);
                            filename = get_nth_img_name(write_img_name, i);
                            cv::imwrite(filename.c_str(), o_img);
                            // std::cerr << "write: " << filename << std::endl;
                            oss << filename << "\t";
                            cropper.dump_metric(i, oss);
                        }
                    }
#pragma omp critical
                    {
                        if (read_recognize_error) {
                            ofs_error << "error: " << src.filename << std::endl;
                            read_recognize_error = false;
                        }
                        std::cerr << oss.str() << std::endl;
                        ofs << oss.str() << std::endl;
                    }
                }
            }
            else if (src.type == 2)
            {
                dest_dirname = src.name + "/../" + get_dirname(src.name) + date_suffix + "/";
#pragma omp for
                for (int i = 0; i < src.dir.filelist.size(); ++i)
                {
                    oss.str("");
                    // http://stackoverflow.com/questions/15033827/multiple-threads-writing-to-stdcout-or-stdcerr

#pragma omp critical
                    {
                        std::cerr << "th" << thread_num << ": " << src.dir.filelist[i] << std::endl;
                    }

                    read_img_name = src.name + "/" + src.dir.filelist[i];
                    write_img_name = dest_dirname + src.dir.filelist[i];
                    // TODO: skip non-image files
                    img_color = cv::imread(read_img_name);
                    if(img_color.data == NULL) {
                        read_recognize_error = true;
                        oss << "Not an image or not supported";
                    }else{
                        dlib::cv_image<dlib::bgr_pixel> cimg(img_color);

                        cropper.detect(cimg);

                        if (0 == cropper.get_num_faces())
                        {
                            read_recognize_error = true;
                            oss << "no face is detected in: " << read_img_name;
                        }
                        for (int i = 0; i < cropper.get_num_faces(); ++i)
                        {
                            cropper.crop_nth(img_color, i, o_img);

                            filename = get_nth_img_name(write_img_name, i);
                            cv::imwrite(filename.c_str(), o_img);
                            // std::cerr << "write: " << filename << std::endl;
                            oss << filename << "\t";
                            cropper.dump_metric(i, oss);
                        }
                    }
#pragma omp critical
                    {
                        if (read_recognize_error)
                        {
                            ofs_error << "error: " << src.filename << std::endl;
                            read_recognize_error = false;
                        }
                        std::cerr << oss.str() << std::endl;
                        // obuf << oss.str() << std::endl;
                        ofs << oss.str() << std::endl << std::flush;
                    }
                }
            }
        }
        // #pragma omp critical
        // {
        //                 ofs << obuf.str();
        // }
    } // end omp parallel

    return 0;
}
