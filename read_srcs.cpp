#include "read_srcs.h" 
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

void readdir_recursive::read_all(std::string dir_str, std::string subdir_str)
{
    DIR *dir;
    struct dirent *ent;
    std::string child_str;
    if (subdir_str != "")
    {
        subdir_str = subdir_str + "/";
    }
    dir_str = dir_str + "/";
    if ((dir = opendir(dir_str.c_str())) != NULL)
    {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL)
        {
            child_str = subdir_str + ent->d_name;
            switch (ent->d_type)
            // http://www.cplusplus.com/forum/beginner/212915/
            {
            // case DT_LNK:
            case DT_REG:
                /* Output file name with directory */
                // printf("file: ");
                filelist.push_back(child_str);
                break;

            case DT_DIR:
                // printf("dir: ");
                /* Scan sub-directory recursively */
                if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0)
                {
                    //                        find_directory(buffer);
                    dirlist.push_back(child_str);
                    read_all(dir_str + ent->d_name, dirlist.back());
                }
                break;

            case DT_LNK:
            default:
                // printf("skip: ");
                /* Ignore device entries */
                /*NOP*/;
            }
            // printf("%s\n", ent->d_name);
        }
        closedir(dir);
    }
    else
    {
        /* could not open directory */
        perror("dir");
        // return 1;
    }
    std::sort(filelist.begin(), filelist.end());
}

void readdir_recursive::dump(std::ostream &os)
{
    os << "files: " << std::endl;
    for (auto i : filelist)
    {
        os << i << std::endl;
    }
    os << "dirs: " << std::endl;
    for (auto i : dirlist)
    {
        os << i << std::endl;
    }
}

void src::read(std::string _name)
{
    name = _name;
    struct stat s;
    if (stat(name.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR)
        {
            type = 2;
            dir.read_all(name, "");
        }
        else if (s.st_mode & S_IFREG)
        {
            type = 1;
            filename = name;
        }
        else
        {
            type = 0;
        }
    }
    else
    {
        // error
        type = 0;
    }
}

void src::dump(std::ostream &os)
{
    os << "name: " << name << std::endl;
    if (type == 1)
    {
        os << "file: " << filename << std::endl;
    }
    else if (type == 2)
    {
        os << "rec_dir:" << std::endl;
        dir.dump(os);
    }
}

void src_set::add(std::string name)
{
    struct src src_obj;
    src_obj.read(name);
    srcs.push_back(src_obj);
}

void src_set::dump(std::ostream &os)
{
    for (auto i : srcs)
    {
        i.dump(os);
    }
}
