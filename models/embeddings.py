import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import os
import pickle

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, texts, batch_size=32):
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # 分词
            encoded_input = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            # 获取embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = model_output.last_hidden_state[:, 0]
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

class FaissIndex:
    def __init__(self, embedding_dim=768, index_path=None):
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts = []
        
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
    
    def add(self, embeddings, texts):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.index.add(embeddings)
        self.texts.extend(texts)
    
    def search(self, query_embedding, k=5):
        if self.index is None or self.index.ntotal == 0:
            return [], []
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx_list in indices:
            batch_results = []
            for idx in idx_list:
                if 0 <= idx < len(self.texts):
                    batch_results.append(self.texts[idx])
                else:
                    batch_results.append("")
            results.append(batch_results)
        
        return distances, results
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.texts", "wb") as f:
            pickle.dump(self.texts, f)
    
    def load(self, path):
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        if os.path.exists(f"{path}.texts"):
            with open(f"{path}.texts", "rb") as f:
                self.texts = pickle.load(f)