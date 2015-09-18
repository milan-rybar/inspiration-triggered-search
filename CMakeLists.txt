cmake_minimum_required(VERSION 2.8.4)
project(its)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${its_BINARY_DIR}")

# libraries used as shared libraries
add_subdirectory(libs)

# C++11 and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")# -march=native")

# source codes
set(SOURCE_DIRECTORY "${its_SOURCE_DIR}/src")

set(SOURCE_FILES
	${SOURCE_DIRECTORY}/main.cpp)

# directory for libraries
include_directories("${its_SOURCE_DIR}/libs")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# OpenCV dependency
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Python dependency
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# NumPy dependency
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

# Boost dependency
find_package(Boost REQUIRED COMPONENTS python serialization)
include_directories(${Boost_INCLUDE_DIRS})

#add_executable(its ${SOURCE_FILES})
# compile as library
add_library(its SHARED ${SOURCE_FILES})

SET_TARGET_PROPERTIES(its PROPERTIES
	OUTPUT_NAME its
	PREFIX ""
)

target_link_libraries(its
	${OpenCV_LIBS}
	${PYTHON_LIBRARIES}
	${Boost_LIBRARIES}
	MultiNEAT)

set(CMAKE_BUILD_TYPE Release)
