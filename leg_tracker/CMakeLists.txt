cmake_minimum_required(VERSION 3.8)
project(leg_tracker)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(leg_tracker_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV 4 REQUIRED)

ament_python_install_package(${PROJECT_NAME})


include_directories(include ${rclcpp_INCLUDE_DIRS})
link_directories(${rclcpp_LIBRARY_DIRS})

add_executable(
  detect_leg_clusters
  src/detect_leg_clusters.cpp
  src/laser_processor.cpp
  src/cluster_features.cpp
)



install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_target_dependencies(
  detect_leg_clusters
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  nav_msgs
  leg_tracker_interfaces
  tf2
  tf2_geometry_msgs
  tf2_ros
  OpenCV
)

add_executable(
  local_occupancy_grid_mapping
  src/local_occupancy_grid_mapping.cpp
  src/laser_processor.cpp
)

ament_target_dependencies(
  local_occupancy_grid_mapping 
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  nav_msgs
  leg_tracker_interfaces
  tf2
  tf2_geometry_msgs
  tf2_ros
  OpenCV
)

install(
  PROGRAMS
  leg_tracker/joint_leg_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS
  detect_leg_clusters
  local_occupancy_grid_mapping
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h"
)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_export_dependencies(rclcpp)
ament_export_dependencies(rclpy)
ament_export_include_directories(include ${rclcpp_INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${rclcpp_LIBRARIES})

ament_package()