#ifndef READ_SRCS_H_
#define READ_SRCS_H_

#include <vector>
#include <iostream>
// desktop.ini

struct readdir_recursive
{
    std::vector<std::string> filelist;
    std::vector<std::string> dirlist;
    // std::string subdir_str;
    void read_all(std::string dir_str, std::string subdir_str);
    void dump(std::ostream &os);
};

struct src
{
    // 1 file, 2 dir
    int type;
    std::string name;
    std::string filename;
    struct readdir_recursive dir;
    void read(std::string _name);
    void dump(std::ostream &os);
};

struct src_set
{
    std::vector<src> srcs;
    void add(std::string name);
    void dump(std::ostream &os);
};

#endif // RWAD_SRCS_H_
