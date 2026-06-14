""" 
@note The purpose of this file is to handle different LLM providers. 
      It includes a base class `LLMProvider` and subclasses for specific providers like Gemini and Ollama. 
      The `generate` method is used to generate text based on prompts provided by the user. 
      This allows the agent to interact with various LLMs, enabling it to provide more accurate and personalized responses. 
""" 

class LLMProvider:
    """ A class that provides a way to interact with LLMs. It includes methods for generating text based on prompts. """
    def generate(self, prompt: str) -> str:
        pass
        

class GeminiProvider(LLMProvider):
    def __init__(self):
        self._client = None

    # Function to generate a response based on a prompt
    def generate(self, prompt: str) -> str:
        if self._client is None:
            key = input("Enter your Gemini API key: ").strip()
            from google import genai
            self._client = genai.Client(api_key=key)
        response = self._client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text

class OllamaProvider(LLMProvider):
    """Covers Qwen or any model served via Ollama's OpenAI-compatible endpoint."""
    def __init__(self, model: str = "qwen2.5", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    # Function to generate a response based on a prompt
    def generate(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        resp.raise_for_status()
        return resp.json()["response"]

class CustomAPIProvider(LLMProvider):
    """For any OpenAI-compatible endpoint."""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
    
    #Fumction to generate a response based on a prompt
    def generate(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# agent/providers.py

def get_llm_provider(cfg: dict) -> LLMProvider:
    """get_llm_provider Generates an appropriate LLMProvider based on the configuration provided.

    Args:
        cfg (dict): The configuration dictionary containing the LLM settings.

    Raises:
        ValueError: If the configuration does not specify a valid LLM provider.

    Returns:
        LLMProvider: The appropriate LLMProvider instance.
    """
    llm_cfg = cfg.get("llm", {})
    kind = llm_cfg.get("provider", "gemini") # default to gemini if not specified

    # elif loop to find the appropriate provider
    if kind == "gemini":
        return GeminiProvider()
    elif kind == "ollama":
        return OllamaProvider(
            model=llm_cfg.get("model", "qwen2.5"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434")
        )
    elif kind == "custom":
        return CustomAPIProvider(
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg.get("api_key") or input("Enter API key: ").strip(),
            model=llm_cfg["model"]
        )
    else:
        raise ValueError(f"Unknown LLM provider: {kind!r}")