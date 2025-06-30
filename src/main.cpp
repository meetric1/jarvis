#include <math.h>
#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>
#include <cctype>

#include "raylib.h"
#include "raymath.h"
#include "portaudio.h"
#include "whisper_interface.h"
#include "llama_interface.h"

#define SAMPLE_RATE WHISPER_SAMPLE_RATE     // not 44100?
#define FRAMES_PER_BUFFER 128
#define FRAMES_BACKUP 50 // Frames cached before talking

#define SPEAKING_THRESHOLD_OFF 0.1
#define SPEAKING_THRESHOLD_ON 0.01
#define SPEAKING_TIME 0.5
#define SPEAKING_TIME_MIN 1.0
#define SPEAKING_TIME_MAX 10    // TODO: crashes if we speak for more than this amount

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
    std::vector<float> pcm;
    std::vector<float> pcm_backup;
    std::vector<float> blocks;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> curtime;

    int frame = 0;
    bool recording = false;

    WhisperInterface whisper_interface;
    LlamaInterface llama_interface;

    StreamData()
        : pcm(SAMPLE_RATE * FRAMES_PER_BUFFER * SPEAKING_TIME_MAX, 0.0f)
        , pcm_backup(SAMPLE_RATE * FRAMES_PER_BUFFER * FRAMES_BACKUP, 0.0f)
        , blocks(FRAMES_PER_BUFFER, 0.0f)
    {};
};

/*void parse_text(std::string text) {
    // tolower
    for (char& ch : text) ch = std::tolower(ch);

    size_t pos = text.find("jarvis");
    if (pos == std::string::npos) return;    // no jarvis :(

    printf("Jarvis at your command: <%s>\n", text.substr(pos + 7).c_str());
}*/

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
        float slice = std::abs(input[i]);
        stream_data->blocks[i] = slice;

        largest = std::max(largest, slice);
    }

    // Start recording
    auto curtime = get_curtime();
    if(!stream_data->recording && largest > SPEAKING_THRESHOLD_OFF) {
        stream_data->start_time = curtime;
        stream_data->curtime = curtime;
        stream_data->recording = true;

        // Load a few frames before we started talking for a bit of extra context
        // Remember we're using a cyclic array so we need to index accordingly
        int till_end = (stream_data->frame % FRAMES_BACKUP);
        int backup_offset = till_end * FRAMES_PER_BUFFER;
        memcpy(
            stream_data->pcm.data(), 
            stream_data->pcm_backup.data() + backup_offset, 
            sizeof(float) * FRAMES_PER_BUFFER * till_end
        );

        memcpy(
            stream_data->pcm.data() + backup_offset, 
            stream_data->pcm_backup.data(), 
            sizeof(float) * FRAMES_PER_BUFFER * (FRAMES_BACKUP - till_end)
        );

        stream_data->frame = FRAMES_BACKUP;
    }

    // custom, simple VAD
    if (stream_data->recording) {
        // Copy current input stream into pcm buffer
        memcpy(
            stream_data->pcm.data() + stream_data->frame * FRAMES_PER_BUFFER, 
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
                if (cast_curtime(curtime - stream_data->start_time) < SPEAKING_TIME_MIN) return 0;

                // we're not talking audio/text processing
                std::string text = stream_data->whisper_interface.process(stream_data->pcm.data(), stream_data->frame * FRAMES_PER_BUFFER);
                stream_data->llama_interface.process(text); //parse_text(text);
            }
        }
    } else {
        memcpy(
            stream_data->pcm_backup.data() + (stream_data->frame % FRAMES_BACKUP) * FRAMES_PER_BUFFER, 
            input_buffer, 
            sizeof(float) * FRAMES_PER_BUFFER
        );
    }
    
    stream_data->frame++;

    return 0;
}

int main() {
    // Interfaces
    ggml_backend_load_all();

    WhisperInterface whisper_interface = WhisperInterface();
    if (!whisper_interface.allocate()) {
        printf("Whisper failed to load\n");
        return 1;
    };

    LlamaInterface llama_interface = LlamaInterface();
    if (!llama_interface.allocate()) {
        printf("Llama failed to load\n");
        return 1;
    };

    // Audio processing
    StreamData stream_data = StreamData();
    stream_data.whisper_interface = whisper_interface;
    stream_data.llama_interface = llama_interface;

    // Audio device setup
    assert_pa(Pa_Initialize());
    int device_id = Pa_GetDefaultInputDevice();
    if (device_id < 0) {
        printf("[PA ERROR]: Couldn't find an audio input\n");
        return 1;
    } else {
        printf("Using Audio Input %d (%s)\n", device_id, Pa_GetDeviceInfo(device_id)->name);
    }

    PaStreamParameters audio_params = PaStreamParameters();
    audio_params.channelCount = 1;
    audio_params.device = device_id;
    audio_params.sampleFormat = paFloat32;
    audio_params.suggestedLatency = Pa_GetDeviceInfo(device_id)->defaultLowInputLatency;

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

    // Raylib visual stuff
    Camera3D camera = Camera3D();
    camera.position = Vector3{0.0, 0.0, 0.0};
    camera.target = Vector3{1.0f, 0.0f, 0.0f};
    camera.up = Vector3{0.0f, 0.0f, 1.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Font arial = LoadFont("resources/arial.ttf");
    Model fuck_you = LoadModel("resources/text.obj");

    int anim_frames = 0;
    Image jarvis_image = LoadImageAnim("resources/jarvis.gif", &anim_frames);
    Texture2D jarvis_texture = LoadTextureFromImage(jarvis_image);

    float curtime = 0.0;
    while (!WindowShouldClose()) {
        UpdateTexture(jarvis_texture, ((unsigned char*)jarvis_image.data) + jarvis_image.width * jarvis_image.height * 4 * ((int)(curtime * 30) % anim_frames));

        BeginDrawing();
            ClearBackground(stream_data.recording ? Color{20, 200, 20} : Color{200, 20, 20});

            // 'Fuck you nvidia' text
            curtime += GetFrameTime();
            BeginMode3D(camera);
            fuck_you.transform = MatrixRotateXYZ(Vector3{0, 0, (float)(1.0f - fmod(curtime, 1)) * -PI + PI / 2});
            DrawModelEx(fuck_you, Vector3{24, -13, 8}, Vector3{0, 0, 1}, 0, Vector3{0.5, 1, 1}, WHITE);
            EndMode3D();

            int scrw = GetScreenWidth();
            int scrh = GetScreenHeight();

            // jarvis
            Rectangle src_rec = { 0.0f, 0.0f, (float)jarvis_texture.width, (float)jarvis_texture.height };
            Rectangle dst_rec = { 0.0f, 0.0f, (float)scrw / 3.0f, (float)scrh / 3.0f};
            DrawTexturePro(jarvis_texture, src_rec, dst_rec, Vector2{0.0f, (float)scrh * 2.0f / -3.0f}, 0, WHITE);

            // Audio visualization
            for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
                float percent = (float)i / FRAMES_PER_BUFFER;
                
                DrawRectangle(
                    percent * scrw, 
                    scrh * (1.0 - std::abs(stream_data.blocks[i])),
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

            DrawTextEx(arial, time_text.c_str(),       Vector2{0,  0}, 25, 2, WHITE);
            DrawTextEx(arial, start_time_text.c_str(), Vector2{0, 30}, 25, 2, WHITE);
            DrawTextEx(arial, stream_data.whisper_interface.last_text.c_str(), Vector2{0, 100}, 25, 2, WHITE);
            std::string wrapped = stream_data.llama_interface.last_text;
            int i = 0;
            for (char& ch : wrapped) {
                i++;
                if (ch == ' ' && i > scrw / 15) {
                    i = 0;
                    ch = '\n';
                }
            }
            DrawTextEx(arial, wrapped.c_str(), Vector2{0, 150}, 25, 2, WHITE);
        EndDrawing();
    }

    // Cleanup memory
    UnloadFont(arial);
    UnloadModel(fuck_you);
    UnloadTexture(jarvis_texture);
    UnloadImage(jarvis_image);
    CloseWindow();

    assert_pa(Pa_StopStream(audio_stream));
    assert_pa(Pa_CloseStream(audio_stream));
    assert_pa(Pa_Terminate());

    whisper_interface.deallocate();
    llama_interface.deallocate();

    printf("Closed Successfully\n");

    return 0;
}