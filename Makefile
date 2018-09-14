CXX = g++

CXXFLAGS = -pipe -std=gnu++11 -fopenmp -O2 -g
#INCPATH = -I/home/tako/progs/include
#LIBPATH = -L/home/tako/progs/lib
#INCPATH = -I ~/dlib_install/include
#LIBPATH = -L ~/dlib_install/lib
#INCPATH = -I ~/lprogs/dlib_install/include -I ~/lprogs/opencv-2.4.12.2/include
#LIBPATH = -L ~/lprogs/dlib_install/lib -L ~/lprogs/opencv-2.4.12.2/lib
#LIBRARY = -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_core -lopencv_highgui -ldlib

INCPATH = -I ~/lprogs/dlib_install/include -I ~/lprogs/opencv-3.4.1/include
LIBPATH = -L ~/lprogs/dlib_install/lib -L ~/lprogs/opencv-3.4.1/lib
LIBRARY = -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -ldlib

TARGET = hpe_4 multi_face_cropper single_face_cropper thread_test face_cropper_dnd
OBJS = multi_face_cropper.o face_cropper.o hpe_3.o read_srcs.o functions.o

.PHONY = test

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) -c $^

all: $(OBJS) $(TARGET) affine_forever

$(TARGET): % : %.o face_cropper.o read_srcs.o functions.o
	$(CXX) $(CXXFLAGS) $(LIBPATH) $^ $(LIBRARY) -o $*

affine_forever: %: %.o
	$(CXX) $(CXXFLAGS) $(LIBPATH) $^ $(LIBRARY) -o $*

test: all
	# time ./face_cropper_dnd to_recognize
	bash -c "time ./affine_forever"
