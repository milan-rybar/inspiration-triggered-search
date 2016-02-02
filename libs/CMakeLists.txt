cmake_minimum_required(VERSION 2.8.4)
project(MultiNEAT)

# C++11 and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")# -march=native")

# source codes
set(SOURCE_DIRECTORY "${MultiNEAT_SOURCE_DIR}/MultiNEAT/lib")

set(SOURCE_FILES
	${SOURCE_DIRECTORY}/Assert.h
	${SOURCE_DIRECTORY}/Genes.h
	${SOURCE_DIRECTORY}/Genome.cpp
	${SOURCE_DIRECTORY}/Genome.h
	${SOURCE_DIRECTORY}/Innovation.cpp
	${SOURCE_DIRECTORY}/Innovation.h
	${SOURCE_DIRECTORY}/NeuralNetwork.cpp
	${SOURCE_DIRECTORY}/NeuralNetwork.h
	${SOURCE_DIRECTORY}/Parameters.cpp
	${SOURCE_DIRECTORY}/Parameters.h
	${SOURCE_DIRECTORY}/PhenotypeBehavior.cpp
	${SOURCE_DIRECTORY}/PhenotypeBehavior.h
	${SOURCE_DIRECTORY}/Population.cpp
	${SOURCE_DIRECTORY}/Population.h
	${SOURCE_DIRECTORY}/PythonBindings.cpp
	${SOURCE_DIRECTORY}/PythonBindings.h
	${SOURCE_DIRECTORY}/Random.cpp
	${SOURCE_DIRECTORY}/Random.h
	${SOURCE_DIRECTORY}/Species.cpp
	${SOURCE_DIRECTORY}/Species.h
	${SOURCE_DIRECTORY}/Substrate.cpp
	${SOURCE_DIRECTORY}/Substrate.h
	${SOURCE_DIRECTORY}/Utils.cpp
	${SOURCE_DIRECTORY}/Utils.h)

# Python dependency
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Boost dependency
find_package(Boost REQUIRED COMPONENTS python serialization)
include_directories(${Boost_INCLUDE_DIRS})

# compile as library
add_library(MultiNEAT SHARED ${SOURCE_FILES})

SET_TARGET_PROPERTIES(MultiNEAT PROPERTIES
	OUTPUT_NAME _MultiNEAT
	PREFIX ""
)

target_link_libraries(MultiNEAT
	${PYTHON_LIBRARIES}
	${Boost_LIBRARIES})

set(CMAKE_BUILD_TYPE Release)
