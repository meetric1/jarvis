#include "whisper.h"

struct WhisperInterface {
	whisper_full_params params;
	whisper_context* context = nullptr;
	std::string last_text = "";

	std::string process(float* pcm, int pcm_samples) {
		int failed = whisper_full(context, params, pcm, pcm_samples);

		if (!failed) {
			last_text = "";
			for (int i = 0; i < whisper_full_n_segments(context); i++) {
				last_text += whisper_full_get_segment_text(context, i);
			}
		} else {
			printf("[WHISPER ERROR]: Failed to process audio [%i]", failed);
		}

		return last_text;
	}

	bool allocate() {
		whisper_context_params init_params = whisper_context_default_params();
		//whisper_init_params.use_gpu = false;

		// https://huggingface.co/ggerganov/whisper.cpp/tree/main
		context = whisper_init_from_file_with_params("resources/ggml-medium.en-q8_0.bin", init_params);
		if (!context) return false;

		// SAMPLING_GREEDY = Less accurate, Less Expensive
		// SAMPLING_BEAM_SEARCH = More accurate, More Expensive
		params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
		//params.print_special    = true;
		//params.print_progress   = true;
		//params.print_realtime   = true;
		//params.print_timestamps = true;
		//params.n_threads = 8;
		params.max_tokens       = 0;    // 32
		params.language         = "en";
		//params.audio_ctx        = 1500;    // https://github.com/ggml-org/whisper.cpp/discussions/297

		return true;
	};

	void deallocate() {
		whisper_free(context);
	};
};