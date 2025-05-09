from models.embeddings import EmbeddingModel, FaissIndex
from models.llm_model import LLMModel
from retrieval.search_engine import GoogleSearch
from retrieval.web_scraper import WebScraper
import config
import os

class RAGRetriever:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 reranker_model_name="gpt-3.5-turbo"):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.reranker = LLMModel(reranker_model_name)
        self.search_engine = GoogleSearch(
            api_key=config.GOOGLE_API_KEY,
            search_engine_id=config.GOOGLE_SEARCH_ENGINE_ID,
            image_search_engine_id=config.GOOGLE_IMAGE_SEARCH_ENGINE_ID,
            cache_dir=os.path.join(config.CACHE_DIR, "search")
        )
        self.web_scraper = WebScraper(cache_dir=os.path.join(config.CACHE_DIR, "pages"))
        self.faiss_index = FaissIndex(
            embedding_dim=config.VECTOR_DIMENSION,
            index_path=config.INDEX_PATH
        )
    
    def retrieve(self, sentences, is_image_query=False):
        self.faiss_index = FaissIndex(embedding_dim=config.VECTOR_DIMENSION)
        
        all_paragraphs = []
        all_metadata = []
        
        for sentence in sentences:
            # 普通网页搜索
            search_results = self.search_engine.search(sentence, num_results=5)
            
            if is_image_query:
                image_results = self.search_engine.search_image(sentence, num_results=5)
                search_results.extend(image_results)
            
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
            reranked_paragraphs = self._rerank(sentences, retrieved_paragraphs)
            
            # 选择TOP_K_RERANK个段落
            evidence = reranked_paragraphs[:config.TOP_K_RERANK]
        else:
            evidence = []
        
        return evidence
    
    def _rerank(self, queries, paragraphs):
        query_text = " ".join(queries)
        
        # 构建重排序提示
        prompt = f"""
        Please rank the following paragraphs based on their relevance to the query.
        The most relevant paragraphs should directly address the claims in the query.
        
        Query: "{query_text}"
        
        Paragraphs:
        {chr(10).join(['[' + str(i+1) + '] ' + p for i, p in enumerate(paragraphs)])}
        
        Return ONLY the numbers of the top {config.TOP_K_RERANK} most relevant paragraphs in order, separated by commas.
        For example: 3,1,5,2,4
        """
        
        system_prompt = "You are a helpful assistant that ranks paragraphs based on their relevance to the query."
        
        response = self.reranker.generate(prompt, system_prompt)
        
        try:
            if ',' in response:
                ranked_indices = [int(idx.strip()) - 1 for idx in response.split(',') if idx.strip().isdigit()]
            else:
                ranked_indices = [int(response.strip()) - 1]
            
            ranked_indices = [idx for idx in ranked_indices if 0 <= idx < len(paragraphs)]
            
            reranked_paragraphs = [paragraphs[idx] for idx in ranked_indices]
            
            remaining_indices = [i for i in range(len(paragraphs)) if i not in ranked_indices]
            reranked_paragraphs.extend([paragraphs[idx] for idx in remaining_indices])
            
            return reranked_paragraphs
        
        except Exception as e:
            print(f"Reranking error: {e}")
            return paragraphs