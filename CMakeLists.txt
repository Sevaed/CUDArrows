cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudarrows LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(GLAD REQUIRED)
find_package(GLFW3 REQUIRED)
 
add_executable(cudarrows
  src/main.cu
  src/chunkupdates.cu
  src/gl/shader.cpp
  src/gl/uniform.cpp
  src/shaders/shader.cpp
  src/camera.cpp
  src/map.cu
  src/base64/base64.cpp
  src/util/reader.cpp
)

target_compile_features(cudarrows PUBLIC cxx_std_11)

set_target_properties(cudarrows
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                 CUDA_ARCHITECTURES OFF)

target_include_directories(cudarrows PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(cudarrows PRIVATE glfw glad::glad ${OPENGL_LIBRARIES})

if(APPLE)
  set_property(TARGET cudarrows
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()