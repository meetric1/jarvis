cmake_minimum_required(VERSION 3.31)
set(CMAKE_BUILD_TYPE Debug)
set(GGML_VULKAN 1)

add_subdirectory(raylib)
add_subdirectory(portaudio)
add_subdirectory(whisper.cpp)

project(jarvis)

file(GLOB SOURCES
	src/*.cpp
)

add_executable(${PROJECT_NAME} 
	${SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
	raylib/src
	portaudio/include
	whisper.cpp/include
	whisper.cpp/ggml/include
	src
)

target_link_libraries(${PROJECT_NAME} 
	raylib
	portaudio
	whisper
)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})