
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)

add_executable(ncnn_face src/ncnn_face.cpp)
target_link_libraries(ncnn_face ncnn ${OpenCV_LIBS})