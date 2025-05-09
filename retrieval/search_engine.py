import requests
import json
import os
import hashlib
import time
from urllib.parse import quote

class GoogleSearch:
    def __init__(self, api_key=None, search_engine_id=None, image_search_engine_id=None, cache_dir=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        self.image_search_engine_id = image_search_engine_id or os.environ.get("GOOGLE_IMAGE_SEARCH_ENGINE_ID")
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, query, is_image=False):
        if not self.cache_dir:
            return None
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        search_type = "image" if is_image else "web"
        
        return os.path.join(self.cache_dir, f"{search_type}_{query_hash}.json")
    
    def search(self, query, num_results=5):
        # 检查缓存
        cache_path = self._get_cache_path(query)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 构建API URL
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            
            # 提取搜索结果
            search_results = []
            if "items" in result:
                for item in result["items"]:
                    search_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            # 保存缓存
            if cache_path:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(search_results, f, ensure_ascii=False)
            
            return search_results
        
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def search_image(self, query, num_results=5):
        # 检查缓存
        cache_path = self._get_cache_path(query, is_image=True)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 构建API URL
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.image_search_engine_id,
            "q": query,
            "searchType": "image",
            "num": min(num_results, 10)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            
            # 提取搜索结果
            search_results = []
            if "items" in result:
                for item in result["items"]:
                    search_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "image_url": item.get("link", ""),
                        "context_url": item.get("image", {}).get("contextLink", "")
                    })
            
            # 保存缓存
            if cache_path:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(search_results, f, ensure_ascii=False)
            
            return search_results
        
        except Exception as e:
            print(f"Image search error: {e}")
            return []