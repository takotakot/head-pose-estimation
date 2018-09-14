#include "functions.h"

#include <sys/stat.h>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <string>

std::string get_dirname(std::string path) {
  while (*path.rbegin() == '/') {
    path.erase(path.size() - 1);
  }
  std::string slash = "/", dot = ".";
  std::string::size_type last_slash, last_dot;
  last_slash = path.find_last_of(slash);
  if (last_slash == std::string::npos) {
    // last_slash = 0;
    return path;
  }
  return path.substr(last_slash + 1);
}

void create_all(std::string date_suffix, struct src_set sst) {
  std::string dest_dirname;
  for (auto src : sst.srcs) {
    if (src.type == 2) {
      dest_dirname =
          src.name + "/../" + get_dirname(src.name) + date_suffix + "/";
      // std::cerr << "mkdir: " << dest_dirname << std::endl;
      if (mkdir(dest_dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | /* rwx */
                                          S_IRGRP | S_IWGRP |
                                          S_IXGRP |                    /* rwx */
                                          S_IROTH | S_IXOTH | S_IXOTH) /* rwx */
          == 0) {
        std::string sub_dirname;
        for (auto dir : src.dir.dirlist) {
          sub_dirname = dest_dirname + dir;
          // std::cerr << "mkdir: " << sub_dirname << std::endl;
          if (mkdir(sub_dirname.c_str(),
                    S_IRUSR | S_IWUSR | S_IXUSR |     /* rwx */
                        S_IRGRP | S_IWGRP | S_IXGRP | /* rwx */
                        S_IROTH | S_IXOTH | S_IXOTH)  /* rwx */
              == 0) {
            // OK
          } else {
            std::cerr << "create error" << std::endl;
          }
        }
      } else {
        std::cerr << "create error" << std::endl;
      }
    }
  }
}

std::string append_date_suffix(std::string read_img_name,
                               std::string date_suffix) {
  std::string slash = "/", dot = ".";
  std::string::size_type last_slash, last_dot;
  std::string appended_filename;

  last_slash = read_img_name.find_last_of(slash);
  if (last_slash == std::string::npos) {
    last_slash = 0;
  }
  std::string filename = read_img_name.substr(last_slash);
  last_dot = filename.find_last_of(dot);
  if (last_dot == std::string::npos) {
    appended_filename = read_img_name + date_suffix + ".jpg";
    return appended_filename;
  }
  appended_filename =
      read_img_name.substr(
          0, std::max(last_slash + last_dot - 1, (std::string::size_type)0)) +
      date_suffix + filename.substr(last_dot);
  return appended_filename;
}

std::string get_date_suffix() {
  time_t timer;
  timer = time(NULL);
  struct tm *timeinfo;
  timeinfo = localtime(&timer);
  char sbuf[256];
  std::strftime(sbuf, 256, "_%F_%H%M", timeinfo);
  return std::string(sbuf);
}

std::string get_nth_img_name(std::string img_name, int n) {
  std::stringstream sst;
  std::string filename = img_name;
  if (n > 0) {
    sst << "_" << n << ".jpg";
    filename += sst.str();
  }
  return filename;
}
