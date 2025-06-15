#include <string>

#include "llama.h"

void llama_error(enum ggml_log_level level, const char* text, void* user_data) {
	if (level >= GGML_LOG_LEVEL_ERROR) {
		printf("%s\n", text);
	}
}

struct LlamaInterface {
	llama_model* model;
	llama_context* context;
	llama_sampler* chain;

	std::string process() {
		return "";
	}

	bool allocate() {
		llama_log_set(llama_error, nullptr);

		llama_model_params init_model_params = llama_model_default_params();
		init_model_params.n_gpu_layers = 99;	// store model on GPU

		// https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF
		model = llama_model_load_from_file("resources/Llama-3.2-1B-Instruct-IQ4_XS.gguf", init_model_params);
		if (!model) return false;

		const llama_vocab* vocab = llama_model_get_vocab(model);
		if (!vocab) return false;

		llama_context_params init_context_params = llama_context_default_params();
		init_context_params.n_ctx = 0;
		//params.n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
		init_context_params.flash_attn = true;
		context = llama_init_from_model(model, init_context_params);
		if (!context) return false;

		llama_sampler_chain_params init_chain_params = llama_sampler_chain_default_params();
		chain = llama_sampler_chain_init(init_chain_params);
		if (!chain) return false;

		// TODO: do the samplers being initiated here need to be freed?
		llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.1f, 1));
    	llama_sampler_chain_add(chain, llama_sampler_init_temp(1.0f));
    	llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

		return true;
	};

	void deallocate() {
		llama_sampler_free(chain);
		llama_free(context);
		llama_model_free(model);
	};
};