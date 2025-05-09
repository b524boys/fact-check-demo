import requests
from bs4 import BeautifulSoup
import hashlib
import os
import json
import time
from urllib.parse import urlparse
import random

class WebScraper:
    def __init__(self, cache_dir=None, user_agents=None):
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 随机切换User-Agent避免被封禁
        self.user_agents = user_agents or [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
    
    def _get_cache_path(self, url):
        if not self.cache_dir:
            return None
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, f"page_{url_hash}.json")
    
    def _get_domain(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def scrape(self, url):
        # 检查缓存
        cache_path = self._get_cache_path(url)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        try:
            time.sleep(random.uniform(1, 3))
            
            # 使用随机User-Agent
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            elements = []
            
            for p in soup.find_all('p'):
                if p.text.strip():
                    elements.append({
                        "type": "p",
                        "text": p.text.strip()
                    })
            
            for i, tag in enumerate(['h1', 'h2', 'h3']):
                for h in soup.find_all(tag):
                    if h.text.strip():
                        elements.append({
                            "type": tag,
                            "text": h.text.strip()
                        })
            
            for div in soup.find_all('div', class_=lambda c: c and any(x in str(c).lower() for x in ['content', 'article', 'main', 'body', 'text'])):
                if div.text.strip() and not any(e['text'] in div.text for e in elements):
                    elements.append({
                        "type": "div",
                        "text": div.text.strip()
                    })
            
            metadata = {
                "url": url,
                "domain": self._get_domain(url),
                "title": soup.title.text.strip() if soup.title else ""
            }
            
            result = {
                "metadata": metadata,
                "elements": elements
            }
            
            if cache_path:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
            
            return result
        
        except Exception as e:
            print(f"Scraping error for {url}: {e}")
            
            # 保存空结果到缓存以避免重复请求失败的URL
            if cache_path:
                empty_result = {
                    "metadata": {"url": url, "domain": self._get_domain(url), "title": ""},
                    "elements": []
                }
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_result, f, ensure_ascii=False)
            
            return {
                "metadata": {"url": url, "domain": self._get_domain(url), "title": ""},
                "elements": []
            }