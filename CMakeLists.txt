CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

PROJECT (detect_tracking)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES("/home/vmt-nuc/Projects/demo/clCaffe/include")
INCLUDE_DIRECTORIES("/home/vmt-nuc/Projects/demo/clCaffe/build/include")

SET(Caffe_LIBRARIES "/home/vmt-nuc/Projects/demo/clCaffe/build/lib/libcaffe.so")

ADD_EXECUTABLE(main yolo_detect.cpp)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Proto_LIBRARIES} -lglog -lprotobuf -lboost_system)
