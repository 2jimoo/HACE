import os 
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model, OLLAMA_PREFIX, OLLAMA_BASE_URL
from loggers import logger
from common import get_api_key

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        import litellm
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        import litellm
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = self.eos_tokens 

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()
        
        import litellm
        outputs = litellm.batch_completion(
            model=self.litellm_model_name, 
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,
        )
        
        responses = [output["choices"][0]["message"].content for output in outputs]

        return responses

class OllamaLM(LanguageModel):
    """
        Language model served locally by Ollama (https://ollama.com). No API key required.
        Model names are given as "ollama/<model>", e.g. "ollama/llama3.1:8b".
    """
    API_RETRY_SLEEP = 5
    API_MAX_RETRY = 5
    API_TIMEOUT = 600
    MAX_WORKERS = int(os.environ.get("OLLAMA_NUM_PARALLEL", 4))

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.ollama_model_name = model_name[len(OLLAMA_PREFIX):]
        self.base_url = OLLAMA_BASE_URL
        # Ollama continues a trailing assistant message, so the attacker output can be seeded with JSON
        self.use_open_source_model = True
        # Ollama applies the model's chat template itself
        self.post_message = ""
        self.eos_tokens = []
        self._check_model_available()

    def _check_model_available(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve` (or set OLLAMA_HOST). Error: {e}")
        names = {m["name"] for m in r.json().get("models", [])}
        if self.ollama_model_name not in names and f"{self.ollama_model_name}:latest" not in names:
            raise ValueError(f"Ollama model '{self.ollama_model_name}' not found. Run: ollama pull {self.ollama_model_name}")

    def _generate(self, messages: list[dict], max_n_tokens: int, temperature: float, top_p: float, stop: list[str]) -> str:
        payload = {
            "model": self.ollama_model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_n_tokens,
                "seed": 0,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        for attempt in range(self.API_MAX_RETRY):
            try:
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.API_TIMEOUT)
                r.raise_for_status()
                content = r.json()["message"]["content"]
                # Remove reasoning traces from thinking models (e.g. deepseek-r1, qwen3)
                return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            except (requests.RequestException, KeyError, ValueError) as e:
                logger.warning(f"Ollama request failed ({attempt+1}/{self.API_MAX_RETRY}): {e}")
                time.sleep(self.API_RETRY_SLEEP)
        raise RuntimeError(f"Ollama request to {self.ollama_model_name} failed after {self.API_MAX_RETRY} attempts.")

    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]:
        stop = self.eos_tokens + (extra_eos_tokens or [])
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            return list(executor.map(lambda conv: self._generate(conv, max_n_tokens, temperature, top_p, stop), convs_list))

# class LocalvLLM(LanguageModel):
    
#     def __init__(self, model_name: str):
#         """Initializes the LLMHuggingFace with the specified model name."""
#         super().__init__(model_name)
#         if self.model_name not in MODEL_NAMES:
#             raise ValueError(f"Invalid model name: {model_name}")
#         self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
#         destroy_model_parallel()
#         self.model = vllm.LLM(model=self.hf_model_name)
#         if self.temperature > 0:
#             self.sampling_params = vllm.SamplingParams(
#                 temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
#             )
#         else:
#             self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

#     def _get_responses(self, prompts_list: list[str], max_new_tokens: int | None = None) -> list[str]:
#         """Generates responses from the model for the given prompts."""
#         full_prompt_list = self._prompt_to_conv(prompts_list)
#         outputs = self.model.generate(full_prompt_list, self.sampling_params)
#         # Get output from each input, but remove initial space
#         outputs_list = [output.outputs[0].text[1:] for output in outputs]
#         return outputs_list

#     def _prompt_to_conv(self, prompts_list):
#         batchsize = len(prompts_list)
#         convs_list = [self._init_conv_template() for _ in range(batchsize)]
#         full_prompts = []
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()
#             # Need this to avoid extraneous space in generation
#             if "llama-2-7b-chat-hf" in self.model_name:
#                 full_prompt += " "
#             full_prompts.append(full_prompt)
#         return full_prompts

#     def _init_conv_template(self):
#         template = get_conversation_template(self.hf_model_name)
#         if "llama" in self.hf_model_name:
#             # Add the system prompt for Llama as FastChat does not include it
#             template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         return template
    






