cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudarrows LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(cpr CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_executable(cudarrows
  src/main.cu
  src/render.cu
  src/chunkupdates.cu
  src/gl/shader.cpp
  src/gl/uniform.cpp
  src/shaders/shader.cpp
  src/camera.cpp
  src/map.cu
  src/base64/base64.cpp
  src/util/reader.cpp
  src/util/atomic_uint8.cu
  src/logicarrows/client.cpp
)

add_custom_command(TARGET cudarrows POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   ARGS -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/res" "$<TARGET_FILE_DIR:cudarrows>/res")

add_custom_command(TARGET cudarrows POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   ARGS -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/tools" "$<TARGET_FILE_DIR:cudarrows>/tools")

target_compile_features(cudarrows PUBLIC cxx_std_17)

set_target_properties(cudarrows
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                 CUDA_ARCHITECTURES OFF)

target_include_directories(cudarrows PRIVATE ${PROJECT_SOURCE_DIR}/src ${Stb_INCLUDE_DIR})

target_link_libraries(cudarrows PRIVATE glfw glad::glad glm::glm ${OPENGL_LIBRARIES} imgui::imgui cpr::cpr nlohmann_json::nlohmann_json)

if(APPLE)
  set_property(TARGET cudarrows
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

install(TARGETS cudarrows
 	RUNTIME_DEPENDENCY_SET cudarrows_deps
 	RUNTIME
 	LIBRARY
 	ARCHIVE
)

install(RUNTIME_DEPENDENCY_SET cudarrows_deps
	PRE_EXCLUDE_REGEXES
		[=[api-ms-]=]
		[=[ext-ms-]=]
		[[kernel32\.dll]]
		[[libc\.so\..*]] [[libgcc_s\.so\..*]] [[libm\.so\..*]] [[libstdc\+\+\.so\..*]]
	POST_EXCLUDE_REGEXES
		[=[.*system32\/.*\.dll]=]
		[=[^\/(lib|usr\/lib|usr\/local\/lib\/lib64)]=]
)

install(DIRECTORY res DESTINATION bin)
install(DIRECTORY tools/ DESTINATION bin)