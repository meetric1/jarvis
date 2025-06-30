# Jarvis

A simple AI assistant using whisper.cpp for audio processing and llama.cpp for text generation. In theory runs on any OS and GPU, though it's only been tested on Windows/Linux

Linux users may need to make their own build.bat but it should be pretty easy to understand.

# Compilation
0. 
Make sure you clone this repo recursively with `--recursive`

It uses `portaudio` for audio input, `whisper.cpp` for audio processing, `llama.cpp` for text generation, and `raylib` for graphics.

1. 
You will need CMake, GCC (MinGW), and (if on windows) the Ninja build system installed. MSVC (cl.exe) gets pissy for some reason and doesn't work.
CMake Installation: `winget install Kitware.CMake`\
Ninja Installation: `winget install Ninja-build.Ninja`\
GCC Installation: https://code.visualstudio.com/docs/cpp/config-mingw \

Alternatively you may be able to use the clang compiler but I haven't tested it

2. 
You will need to download some AI models off the internet. The program currently is hardcoded for these two:\
`ggml-medium.en-q8_0.bin` from: https://huggingface.co/ggerganov/whisper.cpp/tree/main \
`Llama-3.2-1B-Instruct-IQ4_XS` from: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

Put them inside of `resources/`

3. 
Run `build.bat`. It probably will take a few minutes to compile