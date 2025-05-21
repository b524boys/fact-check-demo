import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import os
import pickle

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing EmbeddingModel with device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 验证模型确实在正确的设备上
        print(f"Embedding model is on device: {next(self.model.parameters()).device}")
    
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
            
            # 确保嵌入向量在CPU上进行numpy转换
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

class FaissIndex:
    def __init__(self, embedding_dim=768, index_path=None):
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts = []
        
        if index_path and os.path.exists(f"{index_path}.index"):
            self.load(index_path)
        else:
            # 使用CPU索引，确保与任何设备兼容
            print("Creating new Faiss index (CPU version)")
            self.index = faiss.IndexFlatIP(embedding_dim)
    
    def add(self, embeddings, texts):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 确保输入是numpy数组且类型正确
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        self.index.add(embeddings)
        self.texts.extend(texts)
        print(f"Added {len(texts)} items to the index. Total items: {len(self.texts)}")
    
    def search(self, query_embedding, k=5):
        if self.index is None or self.index.ntotal == 0:
            return [], []
        
        # 确保查询向量是numpy数组且类型正确
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
            
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
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
        print(f"Faiss index saved to {path}.index")
    
    def load(self, path):
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
            print(f"Loaded Faiss index from {path}.index with {self.index.ntotal} vectors")
        if os.path.exists(f"{path}.texts"):
            with open(f"{path}.texts", "rb") as f:
                self.texts = pickle.load(f)
            print(f"Loaded {len(self.texts)} texts from {path}.texts")