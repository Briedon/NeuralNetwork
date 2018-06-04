cmake_minimum_required(VERSION 3.11)
project(parser)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")

add_executable(parser main.cpp parser.h)