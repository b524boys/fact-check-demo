import argparse
from processors.claim_processor import ClaimProcessor
from processors.video_processor import VideoProcessor
from processors.image_processor import ImageProcessor
from retrieval.retriever import RAGRetriever
from fact_checker.verifier import FactVerifier
import json

def parse_args():
    parser = argparse.ArgumentParser(description="事实核查系统")
    parser.add_argument("--claim", type=str, required=True, help="待核查的声明")
    parser.add_argument("--media", type=str, help="媒体文件路径（mp4或jpg）")
    parser.add_argument("--output", type=str, default="result.json", help="输出文件路径")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 处理输入的Claim
    claim_processor = ClaimProcessor()
    claim_sentences = claim_processor.process(args.claim)
    
    media_text = None
    media_sentences = []
    is_image_query = False
    
    # 处理媒体文件（如果有的话）
    if args.media:
        if args.media.lower().endswith('.mp4'):
            # 处理视频
            video_processor = VideoProcessor()
            video_text = video_processor.process(args.media)
            media_text = video_text
            media_sentences = claim_processor.process(video_text)
        elif args.media.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 处理图片
            image_processor = ImageProcessor()
            image_text = image_processor.process(args.media)
            media_text = image_text
            media_sentences = claim_processor.process(image_text)
            is_image_query = True
    
    # 创建检索器
    retriever = RAGRetriever()
    
    # 收集证据
    evidence = retriever.retrieve(claim_sentences + media_sentences, is_image_query)
    
    # 验证事实
    verifier = FactVerifier()
    initial_judgment = verifier.initial_verify(args.claim, media_text)
    final_judgment = verifier.verify_with_evidence(args.claim, media_text, evidence)
    
    # 输出结果
    result = {
        "claim": args.claim,
        "media_text": media_text,
        "initial_judgment": initial_judgment,
        "evidence": evidence,
        "final_judgment": final_judgment
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"事实核查完成。结果已保存至 {args.output}")

if __name__ == "__main__":
    main()