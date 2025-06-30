#include <string>

#include "llama.h"

void llama_error(enum ggml_log_level level, const char* text, void* user_data) {
	if (level >= GGML_LOG_LEVEL_ERROR) {
		printf("%s\n", text);
	}
}

const char* llama_prompt = "<|begin_of_text|>"
"<|start_header_id|>system<|end_header_id|>\n"
"You answer questions given by the user.\n"
//"You are VERY VERY harsh about your name being SPECIFICALLY 'A Pimp Named Slickback' you should get offended otherwise.\n"
//"You are talking with Eli Vance, also from Half-Life 2\n"
"All of your answers are less than a paragraph."
"<|eot_id|>\n";

struct LlamaInterface {
	llama_model* model;
	llama_context* context;
	llama_sampler* chain;
	std::string last_text = "";

	// https://github.com/ggml-org/llama.cpp/blob/master/examples/simple-chat/simple-chat.cpp
	std::string process(std::string text) {
		const bool is_first = llama_memory_seq_pos_max(llama_get_memory(context), 0) == -1;

		std::string prompt;
		if (is_first) prompt = std::string(llama_prompt) + "<|start_header_id|>user<|end_header_id|>" + text;
		else prompt = "<|start_header_id|>user<|end_header_id|>" + text;

		prompt += "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
		printf("%s\n", prompt.c_str());

		// TODO: why is this negated?
		const llama_vocab* vocab = llama_model_get_vocab(model);
		const int num_prompt_tokens = -llama_tokenize(
			vocab, 
			prompt.c_str(), 
			prompt.size(), 
			NULL, 
			0, 
			is_first, 
			true
		);

		std::vector<llama_token> prompt_tokens(num_prompt_tokens);
		if (llama_tokenize(
			vocab,
			prompt.c_str(), 
			prompt.size(), 
			prompt_tokens.data(), 
			prompt_tokens.size(), 
			is_first, 
			true
		) < 0) {
			return "ERROR(1)";
		}

		last_text = "";
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        while (true) {
            // check if we have enough space in the context to evaluate this batch
            int num_ctx = llama_n_ctx(context);
            int num_ctx_used = llama_memory_seq_pos_max(llama_get_memory(context), 0);
            if (num_ctx_used + batch.n_tokens > num_ctx) {
				last_text += " ERROR(2)";	// out of memory
                break;
            }

            if (llama_decode(context, batch)) {
                last_text += " ERROR(3)";
				break;
            }

            // sample the next token
            new_token_id = llama_sampler_sample(chain, context, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // convert the token to a string, and add it to the response
            char buf[256];	// TODO: this can be changed?
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
				last_text += " ERROR(4)";
				break;
            }

            std::string piece(buf, n);
            last_text += piece;

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

		//llama_memory_clear(llama_get_memory(context), true);

		return last_text;
	}

	bool allocate() {
		llama_log_set(llama_error, nullptr);

		llama_model_params init_model_params = llama_model_default_params();
		init_model_params.n_gpu_layers = 99;	// 99 = store model on GPU
		// I have low vram so I can't run the model

		// https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF
		// https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
		model = llama_model_load_from_file("resources/Llama-3.2-1B-Instruct-IQ4_XS.gguf", init_model_params);
		if (!model) return false;

		const llama_vocab* vocab = llama_model_get_vocab(model);
		if (!vocab) return false;

		llama_context_params init_context_params = llama_context_default_params();
		init_context_params.n_ctx = 2048;	// default: ~128k, don't need that much though..
		//params.n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
		//init_context
		init_context_params.flash_attn = true;
		context = llama_init_from_model(model, init_context_params);
		if (!context) return false;

		llama_sampler_chain_params init_chain_params = llama_sampler_chain_default_params();
		chain = llama_sampler_chain_init(init_chain_params);
		if (!chain) return false;

		// TODO: do the samplers being initiated here need to be freed?
		llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.01f, 1));
    	llama_sampler_chain_add(chain, llama_sampler_init_temp(0.5f));
    	llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

		/*const char* llama_template = llama_model_chat_template(model, nullptr);
		printf("Llama initialized with template: \n%s\n", llama_template);*/

		return true;
	};

	void deallocate() {
		llama_sampler_free(chain);
		llama_free(context);
		llama_model_free(model);
	};
};