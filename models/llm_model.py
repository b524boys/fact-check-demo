import requests
import json
import os
from transformers import AutoTokenizer, AutoModel
import torch

class LLMModel:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.is_api = "gpt" in model_name.lower()
        
        if not self.is_api:
            # 加载本地模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
    
    def generate(self, prompt, system_prompt="You are a helpful assistant."):
        if self.is_api:
            return self._api_generate(prompt, system_prompt)
        else:
            return self._local_generate(prompt, system_prompt)
    
    def _api_generate(self, prompt, system_prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}, {response.text}")
    
    def _local_generate(self, prompt, system_prompt):
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取Assistant的回复
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        
        return response