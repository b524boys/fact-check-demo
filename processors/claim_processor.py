from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import os

class ClaimProcessor:
    def __init__(self, model_path="/home/featurize/data/models/deepseek_r1", device=None):
        """
        初始化声明处理器
        
        Args:
            model_path: DeepSeek模型路径
            device: 使用的设备('cuda'或'cpu')
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"初始化ClaimProcessor，使用DeepSeek模型: {model_path}, 设备: {self.device}")
        
        # 验证模型路径是否存在
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
        
        # 加载模型和分词器
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
                
            print(f"DeepSeek模型成功加载于设备: {self.device}")
            
            # 验证模型确实在正确的设备上
            print(f"模型位于设备: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"加载DeepSeek模型时出错: {e}")
            raise
    
    def process(self, claim):
        """
        将声明分解为简单的句子用于搜索
        
        Args:
            claim: 需要验证的声明
            
        Returns:
            list: 简单句子列表，适合搜索引擎查询
        """
        # 构建提示词
        prompt = self._build_prompt(claim)
        
        # 生成回复
        response = self._generate_response(prompt)
        
        # 过滤思考过程
        clean_response = self._filter_thinking_process(response)
        
        # 提取和过滤句子
        print(f"模型原始输出: {clean_response[:200]}...")
        
        sentences = self._extract_sentences(clean_response)
        filtered_sentences = self._filter_invalid_sentences(sentences)
        
        print(f"处理后提取了 {len(filtered_sentences)} 个有效句子")
        
        return filtered_sentences
    
    def _generate_response(self, prompt):
        """
        使用模型生成回复
        
        Args:
            prompt: 提示词
            
        Returns:
            str: 模型生成的回复
        """
        # 确保输入数据在正确的设备上
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 明确将inputs移动到模型所在的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id  # 明确设置pad_token_id
            )
        
        # 确保输出在CPU上进行解码
        outputs = outputs.cpu()
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取模型回复部分（去除原始提示词）
        if response.startswith(prompt):
            response = response[len(prompt):]
        
        return response
    
    def _filter_thinking_process(self, response):
        """
        过滤掉模型思考过程，移除<think>...</think>标签内的内容
        """
        # 移除思考过程
        filtered = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 移除可能的系统指令或其他标记
        filtered = re.sub(r'<.*?>', '', filtered)
        
        return filtered
    
    def _build_prompt(self, claim):
        """
        构建分解声明的提示词
        
        Args:
            claim: 需要分解的声明
            
        Returns:
            str: 提示词
        """
        return f"""You are a helpful assistant designed to help with fact-checking. 

Please break down the given claim into simple, factual sentences that can be independently verified through search engines. 

Important rules:
1. ONLY extract information directly from the given claim
2. Do NOT add any information not present in the original claim
3. Do NOT use any examples or information from this prompt
4. Each sentence should be a simple, verifiable fact
5. Focus on the key assertions in the claim

Guidelines:
- Break complex statements into simple subject-predicate-object sentences
- Preserve all specific details (names, dates, numbers, locations)
- Keep the original meaning and context
- Generate 3-5 most important factual sentences

Claim: {claim}

<think>
I need to analyze this specific claim and break it down into its core factual components.
I must be careful to:
1. Only use information explicitly stated in the claim
2. Not introduce any external knowledge or examples
3. Preserve the exact details mentioned
4. Create simple, searchable sentences
</think>

Simple sentences for verification (based ONLY on the above claim):
"""
    
    def _extract_sentences(self, text):
        """
        从文本中提取句子
        
        Args:
            text: 包含句子的文本
            
        Returns:
            list: 句子列表
        """
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            # 去除行首数字和点
            clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
            
            # 跳过空行
            if not clean_line:
                continue
                
            sentences.append(clean_line)
        
        return sentences
    
    def _filter_invalid_sentences(self, sentences):
        """
        过滤无效的句子，移除思考过程和指令性语言
        
        Args:
            sentences: 待过滤的句子列表
            
        Returns:
            list: 过滤后的句子列表
        """
        filtered = []
        
        for sentence in sentences:
            # 跳过过长的句子（很可能是模型的思考过程）
            if len(sentence) > 150:
                print(f"跳过过长句子: {sentence[:50]}...")
                continue
                
            # 跳过包含指令性语言的句子
            if any(word in sentence.lower() for word in ["i'll", "let me", "first,", "then,", "next,", "finally,"]):
                print(f"跳过指令性句子: {sentence}")
                continue
                
            # 跳过含有模型标记的句子
            if '<' in sentence or '>' in sentence:
                print(f"跳过含标记句子: {sentence}")
                continue
                
            # 确保句子以句号结尾
            if not sentence.endswith('.'):
                sentence = sentence.rstrip(',;:') + '.'
                
            filtered.append(sentence)
        
        return filtered