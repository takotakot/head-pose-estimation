CXX = g++-5

CXXFLAGS = -pipe -std=gnu++11 -O2
#INCPATH = -I/home/tako/progs/include
#LIBPATH = -L/home/tako/progs/lib
INCPATH = -I ~/dlib_install/include
LIBPATH = -L ~/dlib_install/lib
LIBRARY = -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_core -lopencv_highgui -ldlib

TARGET = hpe_4 multi_face_cropper single_face_cropper face_cropper_dnd
OBJS = face_cropper.o hpe_3.o read_srcs.o functions.o

.PHONY = test

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) -c $^

all: $(OBJS) $(TARGET)

$(TARGET): % : %.o face_cropper.o read_srcs.o
	$(CXX) $(CXXFLAGS) $(LIBPATH) $^ $(LIBRARY) -o $*

test: all
	./hpe_4
