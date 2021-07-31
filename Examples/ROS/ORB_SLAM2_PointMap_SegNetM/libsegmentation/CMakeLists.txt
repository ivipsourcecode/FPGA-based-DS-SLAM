cmake_minimum_required (VERSION 2.8)  
project(segmentation)

SET (SRC_LIST libsegmentation.cpp) 
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DFPGA_DEVICE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/home/hero/intelFPGA_pro/17.1/hld/host/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L/home/hero/intelFPGA_pro/17.1/hld/board/a10_ref/linux64/lib -L/home/hero/intelFPGA_pro/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_a10_ref_mmd -lelf")

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../../Examples/ROS/ORB_SLAM2_PointMap_SegNetM/common
/usr/local/include  
/usr/include
) 

add_library(${PROJECT_NAME} SHARED
../common/ocl_util.cpp
../common/timer.cpp
libsegmentation.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS} 
-lboost_system
)

