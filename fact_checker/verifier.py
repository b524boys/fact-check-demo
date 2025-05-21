from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
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

logger = logging.getLogger("fact_verifier")

class FactVerifier:
    def __init__(self, model_path="/home/featurize/data/models/deepseek_r1", device=None):
        """
        Initialize the fact verifier with DeepSeek R1 model
        
        Args:
            model_path: Path to the DeepSeek model
            device: Device to use for inference ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing FactVerifier with DeepSeek model: {model_path}, device: {self.device}")
        
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
                
            logger.info(f"DeepSeek model loaded successfully for fact verification on {self.device}")
            logger.info(f"Model is on device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Error loading DeepSeek model for fact verification: {e}")
            raise
    
    def initial_verify(self, claim, media_text=None):
        """
        Perform initial verification of a claim based on media content
        
        Args:
            claim: The claim to verify
            media_text: Text description of media content (if available)
            
        Returns:
            Dictionary with initial judgment results
        """
        # Generate prompt for initial verification
        if media_text:
            prompt = self._generate_initial_verification_prompt(claim, media_text)
        else:
            prompt = self._generate_initial_verification_prompt_no_media(claim)
        
        # Run inference
        response = self._generate_response(prompt)
        
        # 记录原始响应，便于调试
        logger.info(f"Raw response for initial verification: {response[:500]}...")
        
        # Parse the response - 过滤思考过程
        clean_response = self._filter_thinking_process(response)
        
        logger.info(f"Filtered response for initial verification: {clean_response[:500]}...")
        
        # 尝试多种方法解析响应
        try:
            # 先尝试提取并解析JSON
            result = self._parse_json_response(clean_response)
            
            # 验证结果是否包含必要字段
            if not self._validate_initial_result(result):
                logger.warning(f"Parsed result missing fields: {result}")
                # 尝试使用替代方法提取信息
                result = self._extract_initial_judgment_by_patterns(clean_response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing initial verification response: {e}")
            # 如果JSON解析失败，尝试使用正则表达式提取信息
            try:
                result = self._extract_initial_judgment_by_patterns(clean_response)
                return result
            except Exception as e2:
                logger.error(f"Failed to extract by patterns: {e2}")
                # 保存原始响应，便于进一步调试
                with open("failed_response.txt", "w", encoding="utf-8") as f:
                    f.write(clean_response)
                logger.info("Failed response saved to failed_response.txt")
                
                # 最后使用默认响应
                return {
                    "initial_judgment": "uncertain",
                    "confidence": 0.5,  # 使用0.5而不是0.0，表示不确定而非无证据
                    "reasoning": f"Failed to parse response properly. Original claim: '{claim}'"
                }
    
    def verify_with_evidence(self, claim, media_text=None, evidence=None):
        """
        Verify a claim based on media content and evidence
        
        Args:
            claim: The claim to verify
            media_text: Text description of media content (if available)
            evidence: List of evidence passages
            
        Returns:
            Dictionary with final judgment results
        """
        # 检查证据是否为空
        if not evidence or len(evidence) == 0:
            logger.warning("No evidence provided for verification")
            return {
                "final_judgment": "uncertain",
                "confidence": 0.5,
                "reasoning": "No evidence was provided to verify the claim.",
                "evidence_analysis": []
            }
            
        # 记录证据数量和示例
        logger.info(f"Verifying claim with {len(evidence)} pieces of evidence")
        if evidence and len(evidence) > 0:
            logger.info(f"First evidence example: {evidence[0][:100]}...")
        
        # Generate prompt for verification with evidence
        if media_text:
            prompt = self._generate_evidence_verification_prompt(claim, media_text, evidence)
        else:
            prompt = self._generate_evidence_verification_prompt_no_media(claim, evidence)
        
        # Run inference
        response = self._generate_response(prompt)
        
        # 记录原始响应，便于调试
        logger.info(f"Raw response for evidence verification: {response[:500]}...")
        
        # 过滤思考过程
        clean_response = self._filter_thinking_process(response)
        
        logger.info(f"Filtered response for evidence verification: {clean_response[:500]}...")
        
        # 保存响应以便调试
        with open("verification_response.txt", "w", encoding="utf-8") as f:
            f.write(clean_response)
        logger.info("Verification response saved to verification_response.txt")
        
        try:
            # 先尝试提取并解析JSON
            result = self._parse_json_response(clean_response)
            
            # 验证结果是否包含必要字段
            if not self._validate_final_result(result):
                logger.warning(f"Parsed result missing fields: {result}")
                # 尝试使用替代方法提取信息
                result = self._extract_final_judgment_by_patterns(clean_response, evidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing evidence verification response: {e}")
            # 如果JSON解析失败，尝试使用正则表达式提取信息
            try:
                result = self._extract_final_judgment_by_patterns(clean_response, evidence)
                return result
            except Exception as e2:
                logger.error(f"Failed to extract by patterns: {e2}")
                
                # 最后使用默认响应，但尝试提取部分可用信息
                judgment = "uncertain"
                confidence = 0.5
                reasoning = f"Failed to parse response properly. Original claim: '{claim}'"
                
                # 尝试提取判断结果
                judgment_match = re.search(r'final[_\s]*judgment[\s:"\']*([a-z]+)', clean_response, re.IGNORECASE)
                if judgment_match:
                    judgment = judgment_match.group(1).lower()
                    if judgment not in ["true", "false", "partially_true", "uncertain"]:
                        judgment = "uncertain"
                
                # 构建默认响应
                return {
                    "final_judgment": judgment,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "evidence_analysis": self._create_default_evidence_analysis(evidence)
                }
    
    def _validate_initial_result(self, result):
        """验证初步判断结果是否包含必要字段"""
        if not isinstance(result, dict):
            return False
        return "initial_judgment" in result and "confidence" in result
    
    def _validate_final_result(self, result):
        """验证最终判断结果是否包含必要字段"""
        if not isinstance(result, dict):
            return False
        return "final_judgment" in result and "confidence" in result
    
    def _create_default_evidence_analysis(self, evidence):
        """创建默认的证据分析"""
        analysis = []
        if evidence:
            for i, _ in enumerate(evidence):
                analysis.append({
                    "evidence_id": i + 1,
                    "relevance": "neutral",
                    "explanation": "Failed to properly analyze this evidence."
                })
        return analysis
    
    def _extract_initial_judgment_by_patterns(self, response):
        """使用正则表达式从响应中提取初步判断"""
        result = {
            "initial_judgment": "uncertain",
            "confidence": 0.5,
            "reasoning": "Extracted from partial response."
        }
        
        # 提取判断
        judgment_match = re.search(r'initial[_\s]*judgment[\s:"\']*([a-z_]+)', response, re.IGNORECASE)
        if judgment_match:
            judgment = judgment_match.group(1).lower()
            if judgment in ["true", "false", "partially_true", "uncertain"]:
                result["initial_judgment"] = judgment
            elif "true" in judgment.lower():
                result["initial_judgment"] = "true"
            elif "false" in judgment.lower():
                result["initial_judgment"] = "false"
        
        # 提取置信度
        confidence_match = re.search(r'confidence[\s:"\']*([0-9.]+)', response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if 0 <= confidence <= 1:
                    result["confidence"] = confidence
                elif 0 <= confidence <= 100:  # 如果置信度是0-100范围
                    result["confidence"] = confidence / 100.0
            except:
                pass
        
        # 提取推理
        reasoning_match = re.search(r'reasoning[\s:"\']*([^"\']+)', response, re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # 如果没有找到推理部分，尝试提取响应中最长的段落作为推理
            paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
            if paragraphs:
                longest_paragraph = max(paragraphs, key=len)
                if len(longest_paragraph) > 20:  # 确保段落有一定长度
                    result["reasoning"] = longest_paragraph
        
        return result
    
    def _extract_final_judgment_by_patterns(self, response, evidence=None):
        """使用正则表达式从响应中提取最终判断"""
        result = {
            "final_judgment": "uncertain",
            "confidence": 0.5,
            "reasoning": "Extracted from partial response.",
            "evidence_analysis": []
        }
        
        # 提取判断
        judgment_match = re.search(r'final[_\s]*judgment[\s:"\']*([a-z_]+)', response, re.IGNORECASE)
        if judgment_match:
            judgment = judgment_match.group(1).lower()
            if judgment in ["true", "false", "partially_true", "uncertain"]:
                result["final_judgment"] = judgment
            elif "true" in judgment.lower():
                result["final_judgment"] = "true"
            elif "false" in judgment.lower():
                result["final_judgment"] = "false"
        
        # 提取置信度
        confidence_match = re.search(r'confidence[\s:"\']*([0-9.]+)', response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if 0 <= confidence <= 1:
                    result["confidence"] = confidence
                elif 0 <= confidence <= 100:  # 如果置信度是0-100范围
                    result["confidence"] = confidence / 100.0
            except:
                pass
        
        # 提取推理
        reasoning_match = re.search(r'reasoning[\s:"\']*([^"\']+)', response, re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # 尝试提取分析部分
            analysis_match = re.search(r'analysis[\s:"\']*([^"\']+)', response, re.IGNORECASE)
            if analysis_match:
                result["reasoning"] = analysis_match.group(1).strip()
            else:
                # 如果没有找到推理部分，尝试提取响应中最长的段落作为推理
                paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
                if paragraphs:
                    longest_paragraph = max(paragraphs, key=len)
                    if len(longest_paragraph) > 20:  # 确保段落有一定长度
                        result["reasoning"] = longest_paragraph
        
        # 创建证据分析
        if evidence:
            # 尝试从响应中提取证据分析
            for i, _ in enumerate(evidence):
                evidence_id = i + 1
                relevance = "neutral"
                explanation = "Could not extract specific analysis for this evidence."
                
                # 尝试在响应中查找对这条证据的分析
                evidence_pattern = rf'\[{evidence_id}\][\s\S]*?(supporting|contradicting|neutral)'
                relevance_match = re.search(evidence_pattern, response, re.IGNORECASE)
                if relevance_match:
                    relevance = relevance_match.group(1).lower()
                
                # 尝试提取解释
                explanation_pattern = rf'\[{evidence_id}\][\s\S]*?explanation[\s:"\']*([^"\']+)'
                explanation_match = re.search(explanation_pattern, response, re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                
                result["evidence_analysis"].append({
                    "evidence_id": evidence_id,
                    "relevance": relevance,
                    "explanation": explanation
                })
        
        return result
    
    def _filter_thinking_process(self, response):
        """
        过滤掉模型思考过程，移除<think>...</think>标签内的内容
        """
        # 移除思考过程
        filtered = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 移除可能的系统指令或其他标记
        filtered = re.sub(r'<.*?>', '', filtered)
        
        # 移除任何markdown代码块标记
        filtered = re.sub(r'```(json)?', '', filtered)
        filtered = re.sub(r'```', '', filtered)
        
        return filtered
    
    def _generate_response(self, prompt):
        """Generate a response from the model"""
        try:
            # 确保输入数据在正确的设备上
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 明确将inputs移动到模型所在的设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 使用更激进的生成参数，鼓励模型生成格式良好的JSON
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # 增加最大token数
                    temperature=0.2,      # 降低温度以获得更确定的输出
                    top_p=0.95,
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
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _generate_initial_verification_prompt(self, claim, media_text):
        """Generate a prompt for initial verification with media content"""
        return f"""I need you to initially analyze the following claim based on the provided media content. 
Don't make a final judgment yet, just provide your initial thoughts.

Claim: "{claim}"

Media Content: "{media_text}"

<think>
Let me carefully analyze this claim based on the provided media content:
1. First, I'll understand what the claim is asserting
2. Then, I'll examine the media content to see if it directly supports or contradicts the claim
3. I'll assess the reliability and completeness of the information in the media
4. I'll identify any potential ambiguities or uncertainties
</think>

Provide your initial analysis about whether the claim appears to be true, false, or partially true 
based on the media content alone. If you're uncertain, explain why.

IMPORTANT: Format your output EXACTLY as a JSON object with the following properties:
{{
  "initial_judgment": "true"|"false"|"partially_true"|"uncertain",
  "confidence": [value between 0 and 1],
  "reasoning": "Your explanation here"
}}

Be careful to make sure your JSON is well-formed without any syntax errors. Use double quotes for strings and field names.

JSON Response:
"""
    
    def _generate_initial_verification_prompt_no_media(self, claim):
        """Generate a prompt for initial verification without media content"""
        return f"""I need you to initially analyze the following claim without external evidence. 
Don't make a final judgment yet, just provide your initial thoughts.

Claim: "{claim}"

<think>
Let me carefully analyze what this claim is asserting:
1. I'll break down the key components of the claim
2. I'll consider what I know about this topic from my training data
3. I'll identify any specific factual assertions that can be verified
4. I'll note any ambiguities or potential interpretations
</think>

Provide your initial analysis about whether the claim appears to be true, false, or partially true 
based on your existing knowledge. If you're uncertain, explain why.

IMPORTANT: Format your output EXACTLY as a JSON object with the following properties:
{{
  "initial_judgment": "true"|"false"|"partially_true"|"uncertain",
  "confidence": [value between 0 and 1],
  "reasoning": "Your explanation here"
}}

Be careful to make sure your JSON is well-formed without any syntax errors. Use double quotes for strings and field names.

JSON Response:
"""
    
    def _generate_evidence_verification_prompt(self, claim, media_text, evidence):
        """Generate a prompt for verification with evidence and media content"""
        evidence_text = "\n".join([f"[{i+1}] {e}" for i, e in enumerate(evidence or [])])
        
        return f"""I need you to verify the following claim based on the provided media content and evidence.

Claim: "{claim}"

Media Content: "{media_text}"

Evidence:
{evidence_text}

<think>
Let me systematically analyze this claim:
1. I'll review the claim and media content to understand what's being asserted
2. I'll examine each piece of evidence carefully
3. For each evidence, I'll determine if it supports, contradicts, or is neutral to the claim
4. I'll consider potential reliability issues with any evidence
5. I'll weigh the overall evidence to reach a final conclusion
</think>

Important: Even if you had an initial impression about the claim, your judgment must be based on the provided evidence. Remember that any identified people in the media might be incorrect, so verify their identities using the evidence.

Analyze each piece of evidence and determine whether it supports, contradicts, or is neutral toward the claim.

IMPORTANT: Format your output EXACTLY as a JSON object with the following properties:
{{
  "final_judgment": "true"|"false"|"partially_true"|"uncertain",
  "confidence": [value between 0 and 1],
  "reasoning": "Your detailed explanation of the judgment",
  "evidence_analysis": [
    {{
      "evidence_id": 1,
      "relevance": "supporting"|"contradicting"|"neutral",
      "explanation": "Why this evidence supports/contradicts/is neutral to the claim"
    }},
    // repeat for each piece of evidence
  ]
}}

Be careful to make sure your JSON is well-formed without any syntax errors. Use double quotes for strings and field names.

JSON Response:
"""
    
    def _generate_evidence_verification_prompt_no_media(self, claim, evidence):
        """Generate a prompt for verification with evidence but no media content"""
        evidence_text = "\n".join([f"[{i+1}] {e}" for i, e in enumerate(evidence or [])])
        
        return f"""I need you to verify the following claim based on the provided evidence.

Claim: "{claim}"

Evidence:
{evidence_text}

<think>
Let me systematically analyze this claim:
1. I'll review the claim to understand what's being asserted
2. I'll examine each piece of evidence carefully
3. For each evidence, I'll determine if it supports, contradicts, or is neutral to the claim
4. I'll consider potential reliability issues with any evidence
5. I'll weigh the overall evidence to reach a final conclusion
</think>

Important: Even if you had an initial impression about the claim, your judgment must be based on the provided evidence.

Analyze each piece of evidence and determine whether it supports, contradicts, or is neutral toward the claim.

IMPORTANT: Format your output EXACTLY as a JSON object with the following properties:
{{
  "final_judgment": "true"|"false"|"partially_true"|"uncertain",
  "confidence": [value between 0 and 1],
  "reasoning": "Your detailed explanation of the judgment",
  "evidence_analysis": [
    {{
      "evidence_id": 1,
      "relevance": "supporting"|"contradicting"|"neutral",
      "explanation": "Why this evidence supports/contradicts/is neutral to the claim"
    }},
    // repeat for each piece of evidence
  ]
}}

Be careful to make sure your JSON is well-formed without any syntax errors. Use double quotes for strings and field names.

JSON Response:
"""
    
    def _parse_json_response(self, response):
        """Parse JSON from the model's response"""
        # 首先尝试直接解析（如果响应本身就是JSON）
        try:
            # 删除可能的注释（//开头的行）
            clean_response = re.sub(r'^\s*//.*$', '', response, flags=re.MULTILINE)
            return json.loads(clean_response)
        except:
            pass
        
        # 尝试提取JSON对象
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except:
                # 如果直接解析失败，尝试清理字符串
                # 1. 删除注释
                cleaned_json = re.sub(r'^\s*//.*$', '', json_str, flags=re.MULTILINE)
                # 2. 删除markdown代码块标记
                cleaned_json = re.sub(r'```json|```', '', cleaned_json).strip()
                try:
                    return json.loads(cleaned_json)
                except:
                    # 3. 尝试修复常见的JSON错误
                    # 3.1 修复尾随逗号
                    cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
                    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
                    # 3.2 修复单引号
                    cleaned_json = re.sub(r'\'([^\']*?)\'', r'"\1"', cleaned_json)
                    try:
                        return json.loads(cleaned_json)
                    except Exception as e:
                        logger.error(f"Failed to parse JSON after cleaning: {e}")
                        logger.error(f"Cleaned JSON: {cleaned_json}")
                        # 保存失败的JSON以便调试
                        with open("failed_json.json", "w", encoding="utf-8") as f:
                            f.write(cleaned_json)
                        raise
        
        # 如果没有找到JSON，尝试手动提取键值对
        result = {}
        
        # 提取判断
        judgment_match = re.search(r'(initial_judgment|final_judgment)[\s:"\']*([^",\n]+)', response, re.IGNORECASE)
        if judgment_match:
            result[judgment_match.group(1)] = judgment_match.group(2).strip().lower()
        
        # 提取置信度
        confidence_match = re.search(r'confidence[\s:"\']*([0-9.]+)', response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # 检查置信度范围
                if 0 <= confidence <= 1:
                    result['confidence'] = confidence
                elif 0 <= confidence <= 100:  # 如果模型输出0-100范围
                    result['confidence'] = confidence / 100.0
                else:
                    result['confidence'] = 0.5  # 默认值
            except:
                result['confidence'] = 0.5  # 默认值
        
        # 提取推理
        reasoning_match = re.search(r'reasoning[\s:"\']*([^"\']+)', response, re.IGNORECASE)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        # 如果没有找到任何键值对，抛出异常
        if not result:
            raise ValueError("Could not extract any key-value pairs from the response")
        
        return result