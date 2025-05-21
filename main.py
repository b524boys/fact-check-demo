import argparse
from processors.claim_processor import ClaimProcessor
from processors.video_processor import VideoProcessor
from processors.image_processor import ImageProcessor
from retrieval.retriever import RAGRetriever
from fact_checker.verifier import FactVerifier
from models.qwen_model import QwenModel
import json
import os
import config
import torch
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="Fact-checking system")
    parser.add_argument("--claim", type=str, required=True, help="Claim to verify")
    parser.add_argument("--media", type=str, help="Media file path (mp4 or jpg/png)")
    parser.add_argument("--output", type=str, default="result.json", help="Output file path")
    parser.add_argument("--direct-check", action="store_true", help="Directly check claim with Qwen model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose output")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU mode even if GPU is available")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 显示系统信息
    if args.debug:
        print("\n===== System Information =====")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CPU thread count: {torch.get_num_threads()}")
        print("===============================\n")
    
    # 确定设备
    device = "cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 检查DeepSeek模型路径
    if not os.path.exists(config.DEEPSEEK_MODEL_PATH):
        print(f"ERROR: DeepSeek model not found at {config.DEEPSEEK_MODEL_PATH}")
        print("Please make sure the model is downloaded and available at this location.")
        return
        
    try:
        # 处理输入的声明
        print(f"Processing claim: {args.claim}")
        claim_processor = ClaimProcessor(model_path=config.DEEPSEEK_MODEL_PATH, device=device)
        claim_sentences = claim_processor.process(args.claim)
        print(f"Claim broken down into {len(claim_sentences)} sentences:")
        if args.debug:
            for i, sentence in enumerate(claim_sentences):
                print(f"  {i+1}. {sentence}")
        
        media_text = None
        media_sentences = []
        media_type = None
        
        # 处理媒体文件（如果提供）
        if args.media:
            if not os.path.exists(args.media):
                print(f"ERROR: Media file not found at {args.media}")
                return
                
            # 确定媒体类型
            if args.media.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                # 处理视频
                print(f"Processing video: {args.media}")
                video_processor = VideoProcessor(
                    model_path=config.QWEN_MODEL_PATH,
                    cache_dir=config.MODEL_CACHE_DIR
                )
                video_content = video_processor.process(args.media)
                
                # 获取视觉和音频内容
                visual_text = video_content["visual_content"]
                audio_text = video_content["audio_content"]
                
                if args.debug:
                    print("\n=== Visual Content ===")
                    print(visual_text[:500] + "..." if len(visual_text) > 500 else visual_text)
                    print("\n=== Audio Content ===")
                    print(audio_text[:500] + "..." if len(audio_text) > 500 else audio_text)
                
                # 使用DeepSeek处理文本描述
                print("Breaking down visual content into simple sentences")
                visual_sentences = claim_processor.process(visual_text)
                print("Breaking down audio content into simple sentences")
                audio_sentences = claim_processor.process(audio_text)
                
                # 合并所有句子
                media_sentences = visual_sentences + audio_sentences
                media_text = video_content["full_content"]
                media_type = 'video'
                
                print(f"Video processed: extracted {len(visual_sentences)} visual sentences and {len(audio_sentences)} audio sentences")
                
            elif args.media.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 处理图像
                print(f"Processing image: {args.media}")
                image_processor = ImageProcessor(
                    model_path=config.QWEN_MODEL_PATH,
                    cache_dir=config.MODEL_CACHE_DIR
                )
                image_text = image_processor.process(args.media)
                
                if args.debug:
                    print("\n=== Image Content ===")
                    print(image_text[:500] + "..." if len(image_text) > 500 else image_text)
                
                # 使用DeepSeek处理图像描述
                print("Breaking down image content into simple sentences")
                image_sentences = claim_processor.process(image_text)
                
                media_text = image_text
                media_sentences = image_sentences
                media_type = 'image'
                
                print(f"Image processed: extracted {len(image_sentences)} sentences")
        
        result = {
            "claim": args.claim,
            "media_text": media_text
        }
        
        # 如果设置了direct-check标志，直接使用Qwen模型
        if args.direct_check and args.media:
            print("Directly verifying claim with Qwen model...")
            qwen_model = QwenModel(
                model_path=config.QWEN_MODEL_PATH,
                cache_dir=config.MODEL_CACHE_DIR
            )
            verification_result = qwen_model.verify_claim(args.claim, args.media, media_type)
            
            result["direct_verification"] = {
                "judgment": verification_result,
                "evidence": None
            }
        else:
            # 创建检索器并使用DeepSeek进行重排序
            print("Retrieving evidence...")
            retriever = RAGRetriever(
                embedding_model_name=config.EMBEDDING_MODEL_PATH,
                reranker_model_name=config.DEEPSEEK_MODEL_PATH
            )
            
            # 同时使用声明和媒体句子收集证据
            all_sentences = claim_sentences + media_sentences
            print(f"Using {len(all_sentences)} sentences for retrieval")
            evidence = retriever.retrieve(all_sentences)
            
            if args.debug:
                print("\n=== Retrieved Evidence ===")
                for i, ev in enumerate(evidence):
                    print(f"\nEvidence {i+1}:")
                    print(ev[:200] + "..." if len(ev) > 200 else ev)
            
            # 使用DeepSeek R1验证事实
            print("Verifying claim with DeepSeek R1...")
            verifier = FactVerifier(model_path=config.DEEPSEEK_MODEL_PATH, device=device)
            
            print("Performing initial verification...")
            initial_judgment = verifier.initial_verify(args.claim, media_text)
            
            print("Performing final verification with evidence...")
            final_judgment = verifier.verify_with_evidence(args.claim, media_text, evidence)
            
            # 添加到结果
            result.update({
                "initial_judgment": initial_judgment,
                "evidence": evidence,
                "final_judgment": final_judgment
            })
        
        # 如果输出目录不存在则创建
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 输出结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Fact-checking complete. Result saved to {args.output}")
        
        # 打印结果摘要
        if args.direct_check and args.media:
            print(f"Direct verification result: {result['direct_verification']['judgment']}")
        else:
            print(f"Final judgment: {result['final_judgment'].get('final_judgment', 'unknown')}")
            print(f"Confidence: {result['final_judgment'].get('confidence', 0)}")
            print(f"Number of evidence pieces: {len(result['evidence'])}")
    
    except Exception as e:
        print(f"\n===== ERROR =====")
        print(f"An error occurred: {str(e)}")
        if args.debug:
            print("\nTraceback:")
            traceback.print_exc()
        print("Please check the logs above for more details.")

if __name__ == "__main__":
    main()