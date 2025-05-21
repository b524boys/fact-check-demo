from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os

class DeepSeekReranker:
    """A reranker module based on DeepSeek R1 model for re-ranking retrieved passages"""
    
    def __init__(self, model_path="/home/featurize/data/models/deepseek_r1", device=None):
        """
        Initialize the DeepSeek reranker
        
        Args:
            model_path: Path to the DeepSeek model
            device: Device to use for inference ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing DeepSeek Reranker with model: {model_path}, device: {self.device}")
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 明确指定模型使用的设备
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"  # 让模型自动分配到可用的GPU
                ).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                ).to(self.device).eval()
                
            print(f"DeepSeek model loaded successfully on {self.device}")
            
            # 验证模型确实在正确的设备上
            print(f"Model is on device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"Error loading DeepSeek model: {e}")
            raise
    
    def rerank(self, query: str, paragraphs: List[str], top_k: int = 5) -> List[str]:
        """
        Rerank paragraphs based on relevance to the query
        
        Args:
            query: The query string
            paragraphs: List of paragraphs to rerank
            top_k: Number of top paragraphs to return
        
        Returns:
            List of reranked paragraphs (most relevant first)
        """
        if not paragraphs:
            return []
        
        # Generate reranking prompt
        prompt = self._generate_reranking_prompt(query, paragraphs, top_k)
        
        # Run inference
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,  # Lower temperature for more deterministic ranking
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id  # 明确设置pad_token_id
            )
        
        # 确保输出在CPU上进行解码
        outputs = outputs.cpu()
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the ranked indices from the response
        ranked_indices = self._parse_ranking_response(response, len(paragraphs))
        
        # Return reranked paragraphs
        reranked_paragraphs = []
        for idx in ranked_indices:
            if 0 <= idx < len(paragraphs):
                reranked_paragraphs.append(paragraphs[idx])
                
        # Add any remaining paragraphs that weren't in the ranked indices
        remaining_indices = [i for i in range(len(paragraphs)) if i not in ranked_indices]
        reranked_paragraphs.extend([paragraphs[idx] for idx in remaining_indices])
        
        return reranked_paragraphs
    
    def _generate_reranking_prompt(self, query: str, paragraphs: List[str], top_k: int) -> str:
        """Generate a prompt for reranking paragraphs"""
        paragraphs_text = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(paragraphs)])
        
        prompt = f"""Please rank the following paragraphs based on their relevance to the query.
The most relevant paragraphs should directly address the information in the query.

Query: "{query}"

Paragraphs:
{paragraphs_text}

Return ONLY the numbers of the top {top_k} most relevant paragraphs in order, separated by commas.
For example: 3,1,5,2,4

Ranked paragraphs (most relevant first): """
        
        return prompt
    
    def _parse_ranking_response(self, response: str, num_paragraphs: int) -> List[int]:
        """Parse the model's ranking response to extract ranked indices"""
        # Extract the ranking part after any preamble
        ranking_part = response.split("Ranked paragraphs (most relevant first):")[-1].strip()
        
        # Try to find comma-separated numbers
        if ',' in ranking_part:
            # Find all numbers in the response
            numbers = re.findall(r'\d+', ranking_part)
            indices = [int(num) - 1 for num in numbers if 0 < int(num) <= num_paragraphs]
        else:
            # If no commas, try to get a single number or space-separated numbers
            numbers = re.findall(r'\d+', ranking_part)
            indices = [int(num) - 1 for num in numbers if 0 < int(num) <= num_paragraphs]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = [idx for idx in indices if not (idx in seen or seen.add(idx))]
        
        return unique_indices