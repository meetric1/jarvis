#include "raylib.h"
#include "portaudio.h"
#include "whisper.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>

#define SAMPLE_RATE WHISPER_SAMPLE_RATE     // not 44100?
#define FRAMES_PER_BUFFER 128
#define FRAMES_SPEAK 100 // Frames cached before talking

#define SPEAKING_THRESHOLD_OFF 0.1
#define SPEAKING_THRESHOLD_ON 0.01
#define SPEAKING_TIME 0.5
#define SPEAKING_TIME_MIN 1.0
#define SPEAKING_TIME_MAX 10    // seconds

#define max(a, b) (a > b ? a : b)
#define abs(a) (a > 0 ? a : -a)

Color hsv_to_rgb(float h, float s, float v){
    float r, g, b;

    float i = floor(h * 6.0);
    float f = h * 6.0 - i;
    float p = v * (1.0 - s);
    float q = v * (1.0 - f * s);
    float t = v * (1.0 - (1.0 - f) * s);

    switch ((int)i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return Color{
        (unsigned char)(r * 255), 
        (unsigned char)(g * 255), 
        (unsigned char)(b * 255),
        255
    };
};

void assert_pa(PaError err) {
    if (err == paNoError) return;

    printf("[PA ERROR]: Error Code: %d\n", err);
    exit(1);
}

std::chrono::time_point<std::chrono::high_resolution_clock> get_curtime() {
    return std::chrono::high_resolution_clock::now();
}

double cast_curtime(std::chrono::duration<double, std::ratio<1, 1>> curtime) {
    return curtime.count();
}

struct StreamData {
    float blocks[FRAMES_PER_BUFFER];
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> curtime;
    int frame;
    bool recording;
    whisper_context* whisper_ctx;
    whisper_full_params whisper_params;
    std::string text;
};

// Allocator for whisper (10 seconds)
std::vector<float> pcm(SAMPLE_RATE * FRAMES_PER_BUFFER * SPEAKING_TIME_MAX, 0.0);
std::vector<float> pcm_backup(SAMPLE_RATE * FRAMES_PER_BUFFER * FRAMES_SPEAK, 0.0);

static int audio_callback(
    const void* input_buffer, 
    void* output_buffer,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags,
    void* userdata 
) {
    StreamData* stream_data = (StreamData*)userdata;
    float* input = (float*)input_buffer;

    // Visualization
    float largest = 0.0;
    for (long i = 0; i < frame_count; i++) {
        float slice = input[i];
        stream_data->blocks[i] = slice;

        largest = max(largest, abs(slice));
    }

    // Start recording
    auto curtime = get_curtime();
    if(!stream_data->recording && largest > SPEAKING_THRESHOLD_OFF) {
        stream_data->start_time = curtime;
        stream_data->curtime = curtime;
        stream_data->recording = true;

        // Load a few frames before we started talking for a bit of extra context
        // Remember we're using a cyclic array so we need to index accordingly
        int till_end = (stream_data->frame % FRAMES_SPEAK);
        int backup_offset = till_end * FRAMES_PER_BUFFER;
        memcpy(
            pcm.data(), 
            pcm_backup.data() + backup_offset, 
            sizeof(float) * FRAMES_PER_BUFFER * till_end
        );

        memcpy(
            pcm.data() + backup_offset, 
            pcm_backup.data(), 
            sizeof(float) * FRAMES_PER_BUFFER * (FRAMES_SPEAK - till_end)
        );

        stream_data->frame = FRAMES_SPEAK;
    }

    if (stream_data->recording) {
        // Copy current input stream into pcm buffer
        memcpy(
            pcm.data() + stream_data->frame * FRAMES_PER_BUFFER, 
            input_buffer, 
            sizeof(float) * FRAMES_PER_BUFFER
        );
        
        // Still speaking
        if (largest > SPEAKING_THRESHOLD_ON) {
            stream_data->curtime = curtime;
        } else {
            // We stopped speaking, should we stop recording?
            if (cast_curtime(curtime - stream_data->curtime) > SPEAKING_TIME) {
                stream_data->recording = false;
                
                // We havent spoken for long enough
                if (cast_curtime(curtime - stream_data->start_time) < SPEAKING_TIME_MIN) {
                    stream_data->text = "";
                    return 0;
                }

                /*for (int a = 0; a < stream_data->frame * FRAMES_PER_BUFFER; a++) {
                    for (int b = -10; b <= pcm[a] * 10; b++) {
                        printf("X");
                    }
                    printf("\n");
                };*/

                // Process audio
                int failed = whisper_full(stream_data->whisper_ctx, stream_data->whisper_params, pcm.data(), stream_data->frame * FRAMES_PER_BUFFER);
                if (!failed) {
                    std::string combined = "";
                    for (int i = 0; i < whisper_full_n_segments(stream_data->whisper_ctx); i++) {
                        combined += whisper_full_get_segment_text(stream_data->whisper_ctx, i);
                    }
                    stream_data->text = combined;
                } else {
                    stream_data->text = "Failed to process audio: " + std::to_string(failed);
                }
            }
        }
    } else {
        // Copy current stream into backup
        memcpy(
            pcm_backup.data() + (stream_data->frame % FRAMES_SPEAK) * FRAMES_PER_BUFFER, 
            input_buffer, 
            sizeof(float) * FRAMES_PER_BUFFER
        );
    }
    
    stream_data->frame++;

    return 0;
}

int main() {
    // Whisper.cpp
    ggml_backend_load_all();

    printf("\nggml_backend_dev_count = %i\n\n", ggml_backend_dev_count());
    
    whisper_context_params whisper_init_params = whisper_context_default_params();
    whisper_context* whisper_ctx = whisper_init_from_file_with_params("../resources/ggml-small.en-q8_0.bin", whisper_init_params);
    if (!whisper_ctx) {
        printf("Failed to initialize whisper\n");
        exit(1);
    }

    // SAMPLING_GREEDY = Less accurate, Less Expensive
    // SAMPLING_BEAM_SEARCH = More accurate, More Expensive
    whisper_full_params whisper_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    //whisper_params.print_special    = true;
    //whisper_params.print_progress   = true;
    //whisper_params.print_realtime   = true;
    //whisper_params.print_timestamps = true;
    whisper_params.max_tokens       = 0;    // 32
    whisper_params.language         = "en";
    //whisper_params.audio_ctx        = 1536;    // https://github.com/ggml-org/whisper.cpp/discussions/297


    // Audio device setup
    assert_pa(Pa_Initialize());

    int device_id = Pa_GetDefaultInputDevice();
    if (device_id < 0) {
        printf("[PA ERROR]: Couldn't find an audio input\n");
        exit(1);
    } else {
        printf("Using Audio Input %d (%s)\n", device_id, Pa_GetDeviceInfo(device_id)->name);
    }

    PaStreamParameters audio_params = PaStreamParameters();
    audio_params.channelCount = 1;
    audio_params.device = device_id;
    audio_params.sampleFormat = paFloat32;
    audio_params.suggestedLatency = Pa_GetDeviceInfo(device_id)->defaultLowInputLatency;

    StreamData stream_data = StreamData();
    stream_data.whisper_ctx = whisper_ctx;
    stream_data.whisper_params = whisper_params;

    PaStream* audio_stream;
    assert_pa(Pa_OpenStream(
        &audio_stream,
        &audio_params,
        NULL,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paNoFlag,
        audio_callback,
        (void*)&stream_data
    ));

    assert_pa(Pa_StartStream(audio_stream));
    
    // Audio visualization
    SetTraceLogLevel(LOG_ERROR);
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(800, 450, "Jarvis");
    SetTargetFPS(60);

    Font arial = LoadFont("../resources/arial.ttf");

    while (!WindowShouldClose()) {
        BeginDrawing();
            ClearBackground(stream_data.recording ? Color{20, 200, 20} : Color{200, 20, 20});

            int scrw = GetScreenWidth();
            int scrh = GetScreenHeight();

            // Audio visualization
            for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
                float percent = (float)i / FRAMES_PER_BUFFER;

                DrawRectangle(
                    percent * scrw, 
                    scrh * (1.0 - stream_data.blocks[i]), 
                    (scrw / FRAMES_PER_BUFFER + 1), 
                    scrh, 
                    hsv_to_rgb(percent, 1, 1)
                );
            }

            // Onscreen text
            std::string time_text = "fr: " + std::to_string(stream_data.frame);

            double start_time = cast_curtime(get_curtime() - stream_data.start_time);
            start_time *= 100.0;
            start_time = trunc(start_time);
            start_time /= 100.0;
            std::string start_time_text = "st: " + std::to_string(start_time);

            DrawTextEx(arial, time_text.c_str(),        Vector2{0,   0}, 25, 2, WHITE);
            DrawTextEx(arial, start_time_text.c_str(),  Vector2{0,  30}, 25, 2, WHITE);
            DrawTextEx(arial, stream_data.text.c_str(), Vector2{0, 100}, 25, 2, WHITE);
        EndDrawing();
    }

    // Cleanup memory
    UnloadFont(arial);
    CloseWindow();

    assert_pa(Pa_StopStream(audio_stream));
    assert_pa(Pa_CloseStream(audio_stream));
    assert_pa(Pa_Terminate());

    whisper_free(whisper_ctx);

    printf("Closed Successfully\n");

    return 0;
}