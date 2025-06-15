cmake_minimum_required(VERSION 3.31)
set(CMAKE_BUILD_TYPE Debug)
set(GGML_VULKAN 1)

add_subdirectory(raylib)
add_subdirectory(portaudio)
add_subdirectory(whisper.cpp)
add_subdirectory(llama.cpp)

project(jarvis)

file(GLOB SOURCES
	src/*.cpp
)

add_executable(${PROJECT_NAME}
	${SOURCES}
)

# both whisper.cpp and llama.cpp have ggml
# this means there are 2 branches of the same library
# so im not really sure how thats supposed to work
target_include_directories(${PROJECT_NAME} PRIVATE
	raylib/src
	portaudio/include
	whisper.cpp/include
	llama.cpp/ggml/include
	llama.cpp/include
	src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	raylib
	portaudio
	whisper
	llama
)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})