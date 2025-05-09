from models.llm_model import LLMModel

class ClaimProcessor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = LLMModel(model_name)
    
    def process(self, claim):
        prompt = f"""
        Please break down the following claim into simple subject-predicate-object sentences.
        Each sentence should express only one basic fact.
        
        Claim: "{claim}"
        
        Output each simple sentence on a new line, without numbering.
        """
        
        system_prompt = "You are a helpful assistant that breaks down complex statements into simple factual statements."
        
        response = self.llm.generate(prompt, system_prompt)
        
        # 分割成单独的句子
        sentences = [s.strip() for s in response.split('\n') if s.strip()]
        
        return sentences