CXX = g++-5

CXXFLAGS = -pipe -O -std=gnu++11
INCPATH = -I/home/tako/progs/include
LIBPATH = -L/home/tako/progs/lib
LIBRARY = -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_core -lopencv_highgui -ldlib

TARGET = hpe_3
OBJS = face_cropper.o hpe_3.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) -c $^

all: $(OBJS) $(TARGET)

$(TARGET): % : %.o face_cropper.o
	$(CXX) $(CXXFLAGS) $(LIBPATH) $^ $(LIBRARY) -o $*

