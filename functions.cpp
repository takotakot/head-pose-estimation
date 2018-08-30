#include "functions.h"

void create_all(std::string date_suffix, struct src_set sst)
{
    std::string dest_dirname;
    for (auto src : sst.srcs)
    {
        if (src.type == 2)
        {
            dest_dirname = src.name + "/../" + src.name + date_suffix + "/";
            // std::cerr << "mkdir: " << dest_dirname << std::endl;
            if (mkdir(dest_dirname.c_str(),
                      S_IRUSR | S_IWUSR | S_IXUSR |     /* rwx */
                          S_IRGRP | S_IWGRP | S_IXGRP | /* rwx */
                          S_IROTH | S_IXOTH | S_IXOTH)  /* rwx */
                == 0)
            {
                std::string sub_dirname;
                for (auto dir : src.dir.dirlist)
                {
                    sub_dirname = dest_dirname + dir;
                    // std::cerr << "mkdir: " << sub_dirname << std::endl;
                    if (mkdir(sub_dirname.c_str(),
                              S_IRUSR | S_IWUSR | S_IXUSR |     /* rwx */
                                  S_IRGRP | S_IWGRP | S_IXGRP | /* rwx */
                                  S_IROTH | S_IXOTH | S_IXOTH)  /* rwx */
                        == 0)
                    {
                        // OK
                    }
                    else
                    {
                        std::cerr << "create error" << std::endl;
                    }
                }
            }
            else
            {
                std::cerr << "create error" << std::endl;
            }
        }
    }
}
