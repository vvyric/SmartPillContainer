cmake_minimum_required(VERSION 3.0.2)
project(pill_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  PressureLight.msg
  PillEvent.msg
  SwallowEvent.msg
  ForgetEvent.msg
  PickPlaceEvent.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/pill_detection_node.py
  scripts/pill_detection_simulation_node.py
  scripts/pill_detection_visualization_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)