from models.llm_model import LLMModel
import json

class FactVerifier:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = LLMModel(model_name)
    
    def initial_verify(self, claim, media_text=None):
        # 构建提示
        if media_text:
            prompt = f"""
            I need you to initially analyze the following claim based on the provided media content. 
            Don't make a final judgment yet, just provide your initial thoughts.
            
            Claim: "{claim}"
            
            Media Content: "{media_text}"
            
            Provide your initial analysis about whether the claim appears to be true, false, or partially true 
            based on the media content alone. If you're uncertain, explain why.
            
            Format your response as JSON with the following fields:
            - initial_judgment: "true", "false", "partially_true", or "uncertain"
            - confidence: a number between 0 and 1
            - reasoning: your explanation for the initial judgment
            """
        else:
            prompt = f"""
            I need you to initially analyze the following claim without external evidence. 
            Don't make a final judgment yet, just provide your initial thoughts.
            
            Claim: "{claim}"
            
            Provide your initial analysis about whether the claim appears to be true, false, or partially true 
            based on your existing knowledge. If you're uncertain, explain why.
            
            Format your response as JSON with the following fields:
            - initial_judgment: "true", "false", "partially_true", or "uncertain"
            - confidence: a number between 0 and 1
            - reasoning: your explanation for the initial judgment
            """
        
        system_prompt = "You are a helpful AI assistant tasked with analyzing claims for a fact-checking system."
        
        response = self.llm.generate(prompt, system_prompt)
        
        try:
            return json.loads(response)
        except:
            # 如果无法解析JSON，返回默认响应
            return {
                "initial_judgment": "uncertain",
                "confidence": 0.0,
                "reasoning": "Failed to parse response. Please try again."
            }
    
    def verify_with_evidence(self, claim, media_text=None, evidence=None):
        # 构建提示
        if media_text:
            prompt = f"""
            I need you to verify the following claim based on the provided media content and evidence.
            
            Claim: "{claim}"
            
            Media Content: "{media_text}"
            
            Evidence:
            {chr(10).join([f"[{i+1}] {e}" for i, e in enumerate(evidence or [])])}
            
            Important: Even if you had an initial impression about the claim, your judgment must be based on the provided evidence. Remember that any identified people in the media might be incorrect, so verify their identities using the evidence.
            
            Analyze each piece of evidence and determine whether it supports, contradicts, or is neutral toward the claim.
            
            Format your response as JSON with the following fields:
            - final_judgment: "true", "false", "partially_true", or "uncertain"
            - confidence: a number between 0 and 1
            - reasoning: your detailed explanation of the judgment
            - evidence_analysis: an array of objects, each containing:
                - evidence_id: the number of the evidence [n]
                - relevance: "supporting", "contradicting", or "neutral"
                - explanation: why this evidence supports/contradicts/is neutral to the claim
            """
        else:
            prompt = f"""
            I need you to verify the following claim based on the provided evidence.
            
            Claim: "{claim}"
            
            Evidence:
            {chr(10).join([f"[{i+1}] {e}" for i, e in enumerate(evidence or [])])}
            
            Important: Even if you had an initial impression about the claim, your judgment must be based on the provided evidence.
            
            Analyze each piece of evidence and determine whether it supports, contradicts, or is neutral toward the claim.
            
            Format your response as JSON with the following fields:
            - final_judgment: "true", "false", "partially_true", or "uncertain"
            - confidence: a number between 0 and 1
            - reasoning: your detailed explanation of the judgment
            - evidence_analysis: an array of objects, each containing:
                - evidence_id: the number of the evidence [n]
                - relevance: "supporting", "contradicting", or "neutral"
                - explanation: why this evidence supports/contradicts/is neutral to the claim
            """
        
        system_prompt = "You are a helpful AI assistant tasked with verifying claims for a fact-checking system."
        
        response = self.llm.generate(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            return json.loads(response)
        except:
            return {
                "final_judgment": "uncertain",
                "confidence": 0.0,
                "reasoning": "Failed to parse response. Please try again.",
                "evidence_analysis": []
            }