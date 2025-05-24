from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import os
import logging

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verifier.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("claim_processor")

class OptimizedClaimProcessor:
    def __init__(self, model_path="/home/featurize/data/models/deepseek_r1", device=None):
        """
        初始化优化后的声明处理器
        
        Args:
            model_path: DeepSeek模型路径
            device: 使用的设备('cuda'或'cpu')
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"初始化OptimizedClaimProcessor，使用DeepSeek模型: {model_path}, 设备: {self.device}")
        
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
    
    def process_primary_claim(self, claim):
        """
        处理主要声明，生成核心查询
        
        Args:
            claim: 需要验证的声明
            
        Returns:
            list: 核心查询列表（最多3个）
        """
        prompt = self._build_primary_claim_prompt(claim)
        response = self._generate_response(prompt)
        clean_response = self._filter_thinking_process(response)
        
        print(f"主声明处理 - 模型原始输出: {clean_response[:200]}...")
        
        queries = self._extract_and_validate_queries(clean_response, max_queries=3)
        
        print(f"主声明处理完成，提取了 {len(queries)} 个核心查询")
        
        return queries
    
    def process_media_content(self, content, content_type="media"):
        """
        处理媒体内容，生成补充查询
        
        Args:
            content: 媒体内容描述
            content_type: 内容类型 ("visual", "audio", "image")
            
        Returns:
            list: 补充查询列表（最多2个）
        """
        if not content or len(content.strip()) < 20:
            print(f"{content_type}内容太短，跳过处理")
            return []
            
        prompt = self._build_media_content_prompt(content, content_type)
        response = self._generate_response(prompt)
        clean_response = self._filter_thinking_process(response)
        
        print(f"{content_type}内容处理 - 模型原始输出: {clean_response[:200]}...")
        
        queries = self._extract_and_validate_queries(clean_response, max_queries=2)
        
        print(f"{content_type}内容处理完成，提取了 {len(queries)} 个补充查询")
        
        return queries
    
    def merge_and_deduplicate_queries(self, all_queries):
        """
        合并和去重查询列表
        
        Args:
            all_queries: 所有查询的列表
            
        Returns:
            list: 去重并合并后的查询列表（最多6个）
        """
        if not all_queries:
            return []
        
        # 基础去重
        unique_queries = []
        for query in all_queries:
            if query not in unique_queries:
                unique_queries.append(query)
        
        # 语义相似性检查和合并
        merged_queries = self._merge_similar_queries(unique_queries)
        
        # 按重要性排序并限制数量
        final_queries = self._rank_and_limit_queries(merged_queries, max_queries=6)
        
        print(f"查询合并完成: {len(all_queries)} -> {len(unique_queries)} -> {len(final_queries)}")
        
        return final_queries
    
    def _build_primary_claim_prompt(self, claim):
        """
        构建主声明处理的提示词
        """
        return f"""You are a fact-checking assistant. I need you to analyze the following claim and generate the MOST IMPORTANT search queries to verify it.

IMPORTANT REQUIREMENTS:
1. Generate EXACTLY 2-3 search queries (no more, no less)
2. Each query should be a complete, searchable sentence
3. Focus ONLY on the most critical facts that need verification
4. Avoid overly specific details that might not have search results
5. Make queries broad enough to find relevant information

Claim to analyze: "{claim}"

<think>
I need to identify the core factual assertions in this claim that are most important to verify:
1. What are the main factual claims?
2. What are the key entities, dates, or events mentioned?
3. Which facts are most likely to have reliable sources online?
4. How can I phrase queries to maximize search success?
</think>

Generate EXACTLY 2-3 search queries that focus on the most important facts to verify this claim.

Format your response as a numbered list:
1. [First most important query]
2. [Second most important query]
3. [Third query if necessary]

Search queries:
"""
    
    def _build_media_content_prompt(self, content, content_type):
        """
        构建媒体内容处理的提示词
        """
        return f"""You are a fact-checking assistant. I need you to analyze {content_type} content and generate supplementary search queries for fact-verification.

IMPORTANT REQUIREMENTS:
1. Generate AT MOST 1-2 search queries (prioritize quality over quantity)
2. Only generate queries if the content contains verifiable facts
3. Focus on specific claims that can be fact-checked
4. Avoid vague or general queries
5. Skip queries about subjective descriptions or opinions

{content_type.capitalize()} content to analyze: "{content}"

<think>
I need to analyze this {content_type} content and identify:
1. Are there any specific factual claims that can be verified?
2. Are there mentions of specific people, places, events, or dates?
3. Would these facts benefit from additional verification beyond the main claim?
4. Can I create targeted search queries that would find relevant information?

If the content doesn't contain significant verifiable facts beyond general descriptions, I should generate fewer or no queries.
</think>

Generate 0-2 supplementary search queries based on verifiable facts in this {content_type} content.

Format your response as a numbered list (or "No additional queries needed" if none are warranted):
"""
    
    def _merge_similar_queries(self, queries):
        """
        合并语义相似的查询
        """
        if len(queries) <= 1:
            return queries
        
        merged = []
        used_indices = set()
        
        for i, query1 in enumerate(queries):
            if i in used_indices:
                continue
                
            similar_queries = [query1]
            
            for j, query2 in enumerate(queries[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self._are_queries_similar(query1, query2):
                    similar_queries.append(query2)
                    used_indices.add(j)
            
            # 如果找到相似查询，合并它们
            if len(similar_queries) > 1:
                merged_query = self._combine_queries(similar_queries)
                merged.append(merged_query)
            else:
                merged.append(query1)
            
            used_indices.add(i)
        
        return merged
    
    def _are_queries_similar(self, query1, query2):
        """
        检查两个查询是否语义相似
        """
        # 简单的相似性检查：共同关键词比例
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # 移除停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.6  # 60%相似度阈值
    
    def _combine_queries(self, similar_queries):
        """
        合并相似的查询
        """
        # 选择最长的查询作为基础
        base_query = max(similar_queries, key=len)
        return base_query
    
    def _rank_and_limit_queries(self, queries, max_queries=6):
        """
        对查询进行排序并限制数量
        """
        if len(queries) <= max_queries:
            return queries
        
        # 按长度和复杂度排序（更具体的查询优先）
        scored_queries = []
        for query in queries:
            score = self._calculate_query_score(query)
            scored_queries.append((score, query))
        
        # 按分数降序排序
        scored_queries.sort(key=lambda x: x[0], reverse=True)
        
        # 返回前max_queries个
        return [query for score, query in scored_queries[:max_queries]]
    
    def _calculate_query_score(self, query):
        """
        计算查询的重要性分数
        """
        score = 0
        
        # 长度分数（适中长度更好）
        length = len(query.split())
        if 5 <= length <= 15:
            score += 10
        elif 3 <= length <= 20:
            score += 5
        
        # 包含数字或日期的查询更重要
        if re.search(r'\d+', query):
            score += 15
        
        # 包含专有名词的查询更重要
        if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query):
            score += 10
        
        # 包含重要关键词的查询更重要
        important_keywords = ['president', 'minister', 'company', 'organization', 'event', 'death', 'birth', 'founded', 'established']
        for keyword in important_keywords:
            if keyword.lower() in query.lower():
                score += 5
        
        return score
    
    def _extract_and_validate_queries(self, text, max_queries=3):
        """
        从文本中提取并验证查询
        """
        lines = text.strip().split('\n')
        queries = []
        
        for line in lines:
            # 去除编号和多余空格
            clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
            clean_line = re.sub(r'^[-•]\s*', '', clean_line)
            
            # 跳过空行和太短的行
            if not clean_line or len(clean_line) < 10:
                continue
            
            # 跳过"No additional queries needed"之类的响应
            if any(phrase in clean_line.lower() for phrase in [
                'no additional queries', 'no queries needed', 'no supplementary queries',
                'not necessary', 'no specific queries'
            ]):
                continue
            
            # 验证查询质量
            if self._is_valid_query(clean_line):
                queries.append(clean_line)
                
                if len(queries) >= max_queries:
                    break
        
        return queries
    
    def _is_valid_query(self, query):
        """
        验证查询是否有效
        """
        # 长度检查
        if len(query) < 10 or len(query) > 200:
            return False
        
        # 包含基本词汇
        words = query.lower().split()
        if len(words) < 3:
            return False
        
        # 不能全是问号或感叹号
        if query.count('?') > 2 or query.count('!') > 2:
            return False
        
        # 避免过于通用的查询
        generic_starts = ['what is', 'who is', 'when is', 'where is', 'how is', 'why is']
        if any(query.lower().startswith(start) for start in generic_starts):
            return False
        
        return True
    
    def _generate_response(self, prompt):
        """
        使用模型生成回复
        """
        # 确保输入数据在正确的设备上
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 明确将inputs移动到模型所在的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成回复 - 使用更保守的参数以获得更可控的输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # 减少最大token数
                temperature=0.1,     # 降低温度以获得更确定的输出
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
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
        
        return filtered.strip()