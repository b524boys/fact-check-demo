import re
from bs4 import BeautifulSoup
from models.llm_model import LLMModel

class HTMLCleaner:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = LLMModel(model_name)
    
    def clean_html_text(self, html_text):
        # 使用BeautifulSoup移除HTML标签
        if html_text and '<' in html_text and '>' in html_text:
            soup = BeautifulSoup(html_text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        else:
            text = html_text
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        return text
    
    def organize_elements(self, elements):
        if not elements:
            return []
        
        organized_paragraphs = []
        current_paragraph = []
        
        for element in elements:
            element_type = element.get('type', '')
            text = element.get('text', '').strip()
            
            cleaned_text = self.clean_html_text(text)
            
            if not cleaned_text:
                continue
            
            if element_type in ['h1', 'h2', 'h3']:
                if current_paragraph:
                    organized_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                organized_paragraphs.append(cleaned_text)
            else:
                if len(cleaned_text.split()) > 50:
                    if current_paragraph:
                        organized_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    
                    organized_paragraphs.append(cleaned_text)
                else:
                    current_paragraph.append(cleaned_text)
        
        if current_paragraph:
            organized_paragraphs.append(' '.join(current_paragraph))
        
        return organized_paragraphs