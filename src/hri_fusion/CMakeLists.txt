cmake_minimum_required(VERSION 3.8)
project(hri_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python modules
ament_python_install_package(hri_fusion)

# Install executables
install(PROGRAMS
  hri_fusion/lip_movement_node.py
  hri_fusion/identity_fusion_node.py
  hri_fusion/decision_node.py
  DESTINATION lib/hri_fusion
)

ament_package()
