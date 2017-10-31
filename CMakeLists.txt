3CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

PROJECT (detect_tracking)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES("/home/vmt-nuc/code/clCaffe/include")
INCLUDE_DIRECTORIES("/home/vmt-nuc/code/clCaffe/build/include")

SET(Caffe_LIBRARIES "/home/vmt-nuc/code/clCaffe/build/lib/libcaffe.so")
#SET(Proto_LIBRARIES "/home/vmt-nuc/Applications/clcaffe/caffe/build/lib/libproto.a")
#SET(Caffe_LIBRARIES "${PROJECT_SOURCE_DIR}/libcaffe.so")
#SET(Proto_LIBRARIES "${PROJECT_SOURCE_DIR}/libproto.a")
ADD_EXECUTABLE(main yolo_detect.cpp)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Proto_LIBRARIES} -lglog -lprotobuf -lboost_system)
