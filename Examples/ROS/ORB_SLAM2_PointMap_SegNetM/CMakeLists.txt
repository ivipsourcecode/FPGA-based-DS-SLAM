cmake_minimum_required(VERSION 2.4.6)
include(/opt/ros/indigo/share/ros/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.2.10 REQUIRED)
find_package(Pangolin REQUIRED)
find_package( PCL 1.7 REQUIRED )
find_package( octomap REQUIRED )
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf
  sensor_msgs
  cv_bridge
  image_transport   
)

include_directories(
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${OCTOMAP_INCLUDE_DIRS}
/usr/local/include    
/usr/include
${PROJECT_SOURCE_DIR}/../../../Examples/ROS/ORB_SLAM2_PointMap_SegNetM/common
${USR_DIR}/intelFPGA_pro/17.1/hld/host/include
${PROJECT_SOURCE_DIR}/../../../Examples/ROS/ORB_SLAM2_PointMap_SegNetM/libsegmentation
/opt/ros/indigo/include
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${OCTOMAP_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2_PointMap_SegNetM.so
/opt/ros/indigo/lib/libroscpp.so
/opt/ros/indigo/lib/libroscpp_serialization.so
/opt/ros/indigo/lib/librostime.so
/opt/ros/indigo/lib/librosconsole.so
/opt/ros/indigo/lib/libtf.so 
${PROJECT_SOURCE_DIR}/../../../Examples/ROS/ORB_SLAM2_PointMap_SegNetM/libsegmentation/build/libsegmentation.so
-lboost_system
)

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

rosbuild_add_executable(DS_TUM
rgbd_tum.cc
)

target_link_libraries(DS_TUM
${OpenCV_LIBRARIES}
${LIBS}
)