from models.embeddings import EmbeddingModel, FaissIndex
from models.deepseek_reranker import DeepSeekReranker
from retrieval.search_engine import GoogleSearch
from retrieval.web_scraper import WebScraper
import config
import os

class RAGRetriever:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 reranker_model_name="/home/featurize/data/models/deepseek_r1"):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        
        # Use DeepSeek model for reranking
        self.reranker = DeepSeekReranker(reranker_model_name)
        
        self.search_engine = GoogleSearch(
            api_key=config.GOOGLE_API_KEY,
            search_engine_id=config.GOOGLE_SEARCH_ENGINE_ID,
            cache_dir=os.path.join(config.CACHE_DIR, "search")
        )
        self.web_scraper = WebScraper(cache_dir=os.path.join(config.CACHE_DIR, "pages"))
        self.faiss_index = FaissIndex(
            embedding_dim=config.VECTOR_DIMENSION,
            index_path=config.INDEX_PATH
        )
    
    def retrieve(self, sentences):
        """
        Retrieve relevant paragraphs for the given sentences
        
        Args:
            sentences: List of sentences to retrieve evidence for
            
        Returns:
            List of the most relevant paragraphs
        """
        self.faiss_index = FaissIndex(embedding_dim=config.VECTOR_DIMENSION)
        
        all_paragraphs = []
        all_metadata = []
        
        for sentence in sentences:
            # Regular web search
            search_results = self.search_engine.search(sentence, num_results=5)
            
            for result in search_results:
                url = result['link']
                page_content = self.web_scraper.scrape(url)
                
                for element in page_content['elements']:
                    text = element['text']
                    all_paragraphs.append(text)
                    all_metadata.append({
                        "url": url,
                        "title": page_content['metadata']['title'],
                        "domain": page_content['metadata']['domain']
                    })
        
        if all_paragraphs:
            embeddings = self.embedding_model.encode(all_paragraphs)
            self.faiss_index.add(embeddings, all_paragraphs)
        
        query_embeddings = self.embedding_model.encode(sentences)
        
        retrieved_paragraphs = []
        for i, query_embedding in enumerate(query_embeddings):
            _, results = self.faiss_index.search(query_embedding.reshape(1, -1), k=config.TOP_K_RETRIEVAL)
            
            if results and results[0]:
                retrieved_paragraphs.extend(results[0])
        
        retrieved_paragraphs = list(set(retrieved_paragraphs))
        
        if retrieved_paragraphs:
            # Use DeepSeek for reranking
            query_text = " ".join(sentences)
            reranked_paragraphs = self.reranker.rerank(
                query_text, 
                retrieved_paragraphs,
                top_k=config.TOP_K_RERANK
            )
            
            # Select TOP_K_RERANK paragraphs
            evidence = reranked_paragraphs[:config.TOP_K_RERANK]
        else:
            evidence = []
        
        return evidence