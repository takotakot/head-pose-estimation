#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_
#include "read_srcs.h"

std::string get_dirname(std::string path);
void create_all(std::string date_suffix, struct src_set sst);
std::string append_date_suffix(std::string read_img_name, std::string date_suffix);
std::string get_date_suffix();
std::string get_nth_img_name(std::string img_name, int n);

#endif // FUNCTIONS_H_
