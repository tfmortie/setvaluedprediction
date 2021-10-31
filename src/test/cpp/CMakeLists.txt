cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hsoftmax)
set(CMAKE_PREFIX_PATH "/home/data/tfmortier/.conda/envs/py38/lib/python3.8/site-packages/torch")
find_package(Torch REQUIRED)

add_executable(hsoftmax hsoftmax.cpp)
target_link_libraries(hsoftmax "${TORCH_LIBRARIES}")
set_property(TARGET hsoftmax PROPERTY CXX_STANDARD 14)