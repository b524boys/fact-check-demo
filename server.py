from flask import Flask, request, jsonify, send_file
import os
import json
import uuid
import tempfile
import time
import shutil
from werkzeug.utils import secure_filename
import traceback
from processors.video_processor import VideoProcessor
from processors.image_processor import ImageProcessor
from fact_checker.verifier import FactVerifier
from models.qwen_model import QwenModel
import config
from werkzeug.serving import run_simple
from utils.evidence_enhancer import enhance_evidence
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import torch
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fact_check_server")

app = Flask(__name__)
# 设置最大内容长度为1GB
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 用于存储任务状态和数据
tasks = {}
# 任务锁，用于线程安全
tasks_lock = threading.Lock()

# 创建全局线程池执行器
# 使用线程池可以避免Flask的线程调度问题
task_executor = ThreadPoolExecutor(
    max_workers=8,  # 最多8个工作线程
    thread_name_prefix="fact_check_"
)

# 跟踪活跃的Future对象
active_futures = {}
futures_lock = threading.Lock()

def force_cleanup_gpu_memory():
    """强制清理GPU内存"""
    try:
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 获取当前GPU内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            
            print(f"[内存清理] GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
            logger.info(f"GPU内存清理完成 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
        
        # 强制Python垃圾回收
        collected = gc.collect()
        print(f"[内存清理] Python垃圾回收清理了 {collected} 个对象")
        logger.info(f"Python垃圾回收清理了 {collected} 个对象")
        
    except Exception as e:
        print(f"[内存清理] 清理过程中出错: {e}")
        logger.error(f"GPU内存清理出错: {e}")

def create_and_use_optimized_claim_processor(claim, content_type="primary"):
    """创建并使用优化的ClaimProcessor，用完后立即释放"""
    claim_processor = None
    try:
        print(f"[ClaimProcessor] 开始加载模型...")
        logger.info("创建OptimizedClaimProcessor实例")
        
        # 使用优化后的ClaimProcessor
        from processors.claim_processor import OptimizedClaimProcessor
        claim_processor = OptimizedClaimProcessor(model_path=config.DEEPSEEK_MODEL_PATH)
        
        print(f"[ClaimProcessor] 模型加载完成，开始处理...")
        
        if content_type == "primary":
            result = claim_processor.process_primary_claim(claim)
        else:
            result = claim_processor.process_media_content(claim, content_type)
        
        print(f"[ClaimProcessor] 处理完成，开始清理内存...")
        logger.info("ClaimProcessor处理完成，开始清理")
        
        return result
        
    except Exception as e:
        print(f"[ClaimProcessor] 处理过程中出错: {e}")
        logger.error(f"ClaimProcessor出错: {e}")
        raise
    finally:
        # 显式删除模型实例
        if claim_processor is not None:
            # 删除模型和分词器
            if hasattr(claim_processor, 'model'):
                del claim_processor.model
            if hasattr(claim_processor, 'tokenizer'):
                del claim_processor.tokenizer
            del claim_processor
            
        # 强制清理GPU内存
        force_cleanup_gpu_memory()
        print(f"[OptimizedClaimProcessor] 内存清理完成")

def create_and_use_video_processor(video_path):
    """创建并使用VideoProcessor，用完后立即释放"""
    video_processor = None
    try:
        print(f"[VideoProcessor] 开始加载模型...")
        logger.info("创建VideoProcessor实例")
        
        video_processor = VideoProcessor(
            model_path=config.QWEN_MODEL_PATH,
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        print(f"[VideoProcessor] 模型加载完成，开始处理视频...")
        result = video_processor.process(video_path)
        
        print(f"[VideoProcessor] 视频处理完成，开始清理内存...")
        logger.info("VideoProcessor处理完成，开始清理")
        
        return result
        
    except Exception as e:
        print(f"[VideoProcessor] 处理过程中出错: {e}")
        logger.error(f"VideoProcessor出错: {e}")
        raise
    finally:
        # 显式删除模型实例
        if video_processor is not None:
            # 删除内部的QwenModel
            if hasattr(video_processor, 'model'):
                if hasattr(video_processor.model, 'model'):
                    del video_processor.model.model
                if hasattr(video_processor.model, 'processor'):
                    del video_processor.model.processor
                del video_processor.model
            del video_processor
            
        # 强制清理GPU内存
        force_cleanup_gpu_memory()
        print(f"[VideoProcessor] 内存清理完成")

def create_and_use_image_processor(image_path):
    """创建并使用ImageProcessor，用完后立即释放"""
    image_processor = None
    try:
        print(f"[ImageProcessor] 开始加载模型...")
        logger.info("创建ImageProcessor实例")
        
        image_processor = ImageProcessor(
            model_path=config.QWEN_MODEL_PATH,
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        print(f"[ImageProcessor] 模型加载完成，开始处理图像...")
        result = image_processor.process(image_path)
        
        print(f"[ImageProcessor] 图像处理完成，开始清理内存...")
        logger.info("ImageProcessor处理完成，开始清理")
        
        return result
        
    except Exception as e:
        print(f"[ImageProcessor] 处理过程中出错: {e}")
        logger.error(f"ImageProcessor出错: {e}")
        raise
    finally:
        # 显式删除模型实例
        if image_processor is not None:
            # 删除内部的QwenModel
            if hasattr(image_processor, 'model'):
                if hasattr(image_processor.model, 'model'):
                    del image_processor.model.model
                if hasattr(image_processor.model, 'processor'):
                    del image_processor.model.processor
                del image_processor.model
            del image_processor
            
        # 强制清理GPU内存
        force_cleanup_gpu_memory()
        print(f"[ImageProcessor] 内存清理完成")

def create_and_use_verifier(claim, media_content, evidence=None, is_initial=True):
    """创建并使用FactVerifier，用完后立即释放"""
    verifier = None
    try:
        print(f"[FactVerifier] 开始加载模型...")
        logger.info("创建FactVerifier实例")
        
        verifier = FactVerifier(model_path=config.DEEPSEEK_MODEL_PATH)
        
        print(f"[FactVerifier] 模型加载完成，开始验证...")
        
        if is_initial:
            result = verifier.initial_verify(claim, media_content)
        else:
            result = verifier.verify_with_evidence(claim, media_content, evidence)
        
        print(f"[FactVerifier] 验证完成，开始清理内存...")
        logger.info("FactVerifier处理完成，开始清理")
        
        return result
        
    except Exception as e:
        print(f"[FactVerifier] 处理过程中出错: {e}")
        logger.error(f"FactVerifier出错: {e}")
        raise
    finally:
        # 显式删除模型实例
        if verifier is not None:
            # 删除模型和分词器
            if hasattr(verifier, 'model'):
                del verifier.model
            if hasattr(verifier, 'tokenizer'):
                del verifier.tokenizer
            del verifier
            
        # 强制清理GPU内存
        force_cleanup_gpu_memory()
        print(f"[FactVerifier] 内存清理完成")

def create_and_use_qwen_model(claim, media_path, media_type):
    """创建并使用QwenModel进行直接验证，用完后立即释放"""
    qwen_model = None
    try:
        print(f"[QwenModel] 开始加载模型...")
        logger.info("创建QwenModel实例")
        
        qwen_model = QwenModel(
            model_path=config.QWEN_MODEL_PATH,
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        print(f"[QwenModel] 模型加载完成，开始直接验证...")
        result = qwen_model.verify_claim(claim, media_path, media_type)
        
        print(f"[QwenModel] 直接验证完成，开始清理内存...")
        logger.info("QwenModel处理完成，开始清理")
        
        return result
        
    except Exception as e:
        print(f"[QwenModel] 处理过程中出错: {e}")
        logger.error(f"QwenModel出错: {e}")
        raise
    finally:
        # 显式删除模型实例
        if qwen_model is not None:
            # 删除模型和处理器
            if hasattr(qwen_model, 'model'):
                del qwen_model.model
            if hasattr(qwen_model, 'processor'):
                del qwen_model.processor
            del qwen_model
            
        # 强制清理GPU内存
        force_cleanup_gpu_memory()
        print(f"[QwenModel] 内存清理完成")

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/submit_task', methods=['POST'])
def submit_task():
    """接收媒体文件和声明，启动分析任务"""
    try:
        print("收到任务提交请求...")
        logger.info("收到任务提交请求")
        
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务文件夹
        task_folder = os.path.join(UPLOAD_FOLDER, task_id)
        os.makedirs(task_folder, exist_ok=True)
        
        # 获取声明文本
        if 'claim' not in request.form:
            return jsonify({"error": "Missing claim parameter"}), 400
        
        claim = request.form['claim']
        print(f"收到声明: {claim}")
        logger.info(f"任务 {task_id} - 声明: {claim}")
        
        # 接收媒体文件(如果有)
        media_path = None
        media_type = None
        
        if 'media' in request.files:
            media_file = request.files['media']
            if media_file.filename:
                filename = secure_filename(media_file.filename)
                media_path = os.path.join(task_folder, filename)
                print(f"保存媒体文件到: {media_path}")
                media_file.save(media_path)
                
                # 确定媒体类型
                if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    media_type = 'video'
                    print("检测到视频文件")
                elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    media_type = 'image'
                    print("检测到图像文件")
        
        # 创建任务状态记录
        with tasks_lock:
            tasks[task_id] = {
                "status": "processing",
                "claim": claim,
                "media_path": media_path,
                "media_type": media_type,
                "created_at": time.time(),
                "queries": [],  # 将存储需要客户端查询的内容
                "query_results": {},  # 将存储客户端查询的结果
                "complete": False,
                "client_connected": True,  # 客户端连接状态
                "last_activity": time.time()  # 最后活动时间
            }
        
        print(f"任务创建成功，ID: {task_id}")
        logger.info(f"任务 {task_id} 创建成功")
        
        # 使用线程池执行任务处理
        print(f"[{datetime.now()}] 提交任务到线程池: {task_id}")
        future = task_executor.submit(process_task_optimized, task_id)
        
        # 记录Future对象
        with futures_lock:
            active_futures[f"process_{task_id}"] = future
        
        # 添加完成回调
        def on_process_complete(fut):
            try:
                fut.result()
                print(f"[{datetime.now()}] 任务处理完成: {task_id}")
                logger.info(f"任务 {task_id} 处理完成")
            except Exception as e:
                print(f"[{datetime.now()}] 任务处理失败: {task_id}, 错误: {str(e)}")
                logger.error(f"任务 {task_id} 处理失败: {str(e)}")
            finally:
                with futures_lock:
                    active_futures.pop(f"process_{task_id}", None)
        
        future.add_done_callback(on_process_complete)
        
        return jsonify({
            "status": "success", 
            "task_id": task_id,
            "message": "Task submitted successfully"
        })
        
    except Exception as e:
        print(f"提交任务时出错: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"提交任务时出错: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def process_task_optimized(task_id):
    """优化后的任务处理函数，减少查询数量和碎片化"""
    print(f"[{datetime.now()}] [{threading.current_thread().name}] 开始优化处理任务 {task_id}...")
    logger.info(f"[{threading.current_thread().name}] 开始优化处理任务 {task_id}")
    
    with tasks_lock:
        if task_id not in tasks:
            print(f"任务 {task_id} 不存在")
            return
        task = dict(tasks[task_id])  # 创建副本以避免长时间持有锁
    
    try:
        all_queries = []
        
        # 1. 处理主要声明 - 生成2-3个核心查询
        print(f"处理主要声明: {task['claim']}")
        primary_queries = create_and_use_optimized_claim_processor(task['claim'], "primary")
        all_queries.extend(primary_queries)
        
        # 更新任务状态
        with tasks_lock:
            if task_id in tasks and tasks[task_id]["client_connected"]:
                tasks[task_id]["primary_queries"] = primary_queries
                tasks[task_id]["last_activity"] = time.time()
        
        # 2. 处理媒体文件（如果有）
        media_queries = []
        
        if task["media_path"] and task["media_type"]:
            if task["media_type"] == 'video':
                # 处理视频
                print(f"处理视频: {task['media_path']}")
                video_content = create_and_use_video_processor(task['media_path'])
                
                # 只对重要内容生成补充查询
                visual_queries = []
                audio_queries = []
                
                if video_content["visual_content"] and len(video_content["visual_content"]) > 50:
                    visual_queries = create_and_use_optimized_claim_processor(
                        video_content["visual_content"], "visual"
                    )
                
                if video_content["audio_content"] and len(video_content["audio_content"]) > 50:
                    audio_queries = create_and_use_optimized_claim_processor(
                        video_content["audio_content"], "audio"
                    )
                
                media_queries.extend(visual_queries)
                media_queries.extend(audio_queries)
                
                # 更新任务状态
                with tasks_lock:
                    if task_id in tasks and tasks[task_id]["client_connected"]:
                        tasks[task_id]["visual_content"] = video_content["visual_content"]
                        tasks[task_id]["audio_content"] = video_content["audio_content"]
                        tasks[task_id]["media_content"] = video_content["full_content"]
                        tasks[task_id]["visual_queries"] = visual_queries
                        tasks[task_id]["audio_queries"] = audio_queries
                        tasks[task_id]["last_activity"] = time.time()
                
            elif task["media_type"] == 'image':
                # 处理图像
                print(f"处理图像: {task['media_path']}")
                image_text = create_and_use_image_processor(task['media_path'])
                
                # 只对有意义的图像描述生成补充查询
                if image_text and len(image_text) > 50:
                    image_queries = create_and_use_optimized_claim_processor(
                        image_text, "image"
                    )
                    media_queries.extend(image_queries)
                
                # 更新任务状态
                with tasks_lock:
                    if task_id in tasks and tasks[task_id]["client_connected"]:
                        tasks[task_id]["media_content"] = image_text
                        tasks[task_id]["image_queries"] = image_queries if 'image_queries' in locals() else []
                        tasks[task_id]["last_activity"] = time.time()
        
        # 3. 合并所有查询并进行智能去重和优化
        all_queries.extend(media_queries)
        
        # 添加原始claim作为基础查询（如果不在列表中）
        if task['claim'] not in all_queries:
            all_queries.insert(0, task['claim'])
        
        print(f"合并前总查询数: {len(all_queries)}")
        
        # 使用优化的ClaimProcessor进行最终的合并和去重
        optimized_queries = []
        if all_queries:
            # 创建临时处理器实例进行查询优化
            temp_processor = None
            try:
                from processors.claim_processor import OptimizedClaimProcessor
                temp_processor = OptimizedClaimProcessor(model_path=config.DEEPSEEK_MODEL_PATH)
                optimized_queries = temp_processor.merge_and_deduplicate_queries(all_queries)
            except Exception as e:
                print(f"查询优化失败，使用原始查询: {e}")
                # 如果优化失败，进行基础去重
                seen = set()
                optimized_queries = [q for q in all_queries if not (q in seen or seen.add(q))]
                optimized_queries = optimized_queries[:6]  # 限制最多6个
            finally:
                if temp_processor:
                    del temp_processor.model
                    del temp_processor.tokenizer
                    del temp_processor
                    force_cleanup_gpu_memory()
        
        print(f"优化后查询数: {len(optimized_queries)}")
        print(f"最终查询列表: {optimized_queries}")
        
        # 更新任务状态为等待查询
        with tasks_lock:
            if task_id in tasks and tasks[task_id]["client_connected"]:
                tasks[task_id]["queries"] = optimized_queries
                tasks[task_id]["original_query_count"] = len(all_queries)
                tasks[task_id]["optimized_query_count"] = len(optimized_queries)
                tasks[task_id]["status"] = "waiting_for_queries"
                tasks[task_id]["last_activity"] = time.time()
                
                print(f"任务 {task_id} 优化处理完成。")
                print(f"查询优化: {len(all_queries)} -> {len(optimized_queries)} 个查询")
                logger.info(f"任务 {task_id} 优化处理完成，查询数量: {len(all_queries)} -> {len(optimized_queries)}")
            else:
                print(f"任务 {task_id} 客户端已断开连接，停止处理")
                return
        
    except Exception as e:
        print(f"处理任务 {task_id} 时出错: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = str(e)
                tasks[task_id]["traceback"] = traceback.format_exc()
                tasks[task_id]["last_activity"] = time.time()

def verify_results(task_id, use_partial_evidence=False):
    """使用查询结果进行事实验证"""
    print(f"[{datetime.now()}] [{threading.current_thread().name}] 开始验证任务 {task_id}... (部分证据: {use_partial_evidence})")
    logger.info(f"[{threading.current_thread().name}] 开始验证任务 {task_id}")
    
    # 添加进度更新函数
    def update_progress(stage, message):
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["verification_stage"] = stage
                tasks[task_id]["verification_message"] = message
                tasks[task_id]["last_activity"] = time.time()
        print(f"[{datetime.now()}] [验证进度] {task_id}: {stage} - {message}")
        logger.info(f"[验证进度] {task_id}: {stage} - {message}")
    
    update_progress("starting", "开始验证流程")
    
    with tasks_lock:
        if task_id not in tasks:
            print(f"任务 {task_id} 不存在")
            return
        task = dict(tasks[task_id])  # 创建副本
    
    try:
        # 准备证据
        update_progress("preparing_evidence", "正在准备证据")
        all_evidence = []
        for query, results in task["query_results"].items():
            if isinstance(results, list):
                all_evidence.extend(results)
        
        # 去重
        all_evidence = enhance_evidence(list(set(all_evidence)))
        evidence_count = len(all_evidence)
        
        update_progress("evidence_ready", f"已准备 {evidence_count} 条证据")
        
        # 初步验证 - 创建新实例并在完成后清理
        update_progress("initial_verification", "正在进行初步验证")
        initial_judgment = create_and_use_verifier(
            task["claim"], 
            task.get("media_content"),
            is_initial=True
        )
        
        # 基于证据的最终验证
        if evidence_count > 0:
            update_progress("final_verification", "正在进行基于证据的最终验证")
            final_judgment = create_and_use_verifier(
                task["claim"], 
                task.get("media_content"), 
                all_evidence,
                is_initial=False
            )
        else:
            update_progress("no_evidence", "没有外部证据，使用初步验证结果")
            final_judgment = {
                "final_judgment": initial_judgment.get("initial_judgment", "uncertain"),
                "confidence": max(0, initial_judgment.get("confidence", 0.5) - 0.1),
                "reasoning": f"基于媒体内容的判断，未获得外部证据支持。{initial_judgment.get('reasoning', '')}",
                "evidence_analysis": []
            }
        
        # 保存结果
        update_progress("saving_results", "正在保存验证结果")
        result_path = os.path.join(RESULTS_FOLDER, f"{task_id}.json")
        
        result = {
            "task_id": task_id,
            "claim": task["claim"],
            "media_type": task["media_type"],
            "media_content": task.get("media_content"),
            "evidence": all_evidence,
            "evidence_count": evidence_count,
            "used_partial_evidence": use_partial_evidence,
            "initial_judgment": initial_judgment,
            "final_judgment": final_judgment,
            "completed_at": time.time()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 更新任务状态
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["result_path"] = result_path
                tasks[task_id]["complete"] = True
                tasks[task_id]["result"] = result
                tasks[task_id]["last_activity"] = time.time()
                tasks[task_id]["verification_stage"] = "completed"
                tasks[task_id]["verification_message"] = "验证完成"
        
        update_progress("completed", "验证流程完成")
        logger.info(f"任务 {task_id} 验证完成")
        
    except Exception as e:
        print(f"[{datetime.now()}] 验证任务 {task_id} 时出错: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"验证任务 {task_id} 时出错: {str(e)}")
        
        update_progress("error", f"验证失败: {str(e)}")
        
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = str(e)
                tasks[task_id]["traceback"] = traceback.format_exc()
                tasks[task_id]["last_activity"] = time.time()

@app.route('/get_queries/<task_id>', methods=['GET'])
def get_queries(task_id):
    """获取待查询列表，供客户端调用"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        
        # 更新最后活动时间
        task["last_activity"] = time.time()
        
        if task["status"] == "error":
            return jsonify({
                "status": "error",
                "error": task.get("error", "Unknown error"),
                "traceback": task.get("traceback", "")
            })
        
        if task["status"] != "waiting_for_queries":
            return jsonify({
                "status": task["status"],
                "message": "No queries available at this time"
            })
        
        # 返回未处理的查询
        pending_queries = [q for q in task["queries"] if q not in task["query_results"]]
        
        return jsonify({
            "status": "success",
            "task_id": task_id,
            "claim": task.get("claim"),  # 包含claim信息
            "queries": pending_queries,
            "total_queries": len(task["queries"]),
            "completed_queries": len(task["query_results"])
        })

@app.route('/submit_query_results/<task_id>', methods=['POST'])
def submit_query_results(task_id):
    """接收客户端的查询结果，返回确认信息"""
    submit_time = datetime.now()
    print(f"[{submit_time}] 收到查询结果提交请求 - 任务ID: {task_id}")
    logger.info(f"收到查询结果提交请求 - 任务ID: {task_id}")
    
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        
        # 更新最后活动时间
        task["last_activity"] = time.time()
        
        if not task["client_connected"]:
            return jsonify({"error": "Client has disconnected"}), 400
    
    try:
        # 获取查询结果
        query_results = request.json
        
        if not query_results or not isinstance(query_results, dict):
            return jsonify({"error": "Invalid query results format"}), 400
        
        # 记录成功接收的查询
        successfully_received = []
        failed_queries = []
        
        # 逐个处理查询结果，确保幂等性
        with tasks_lock:
            for query, results in query_results.items():
                try:
                    if query not in task["query_results"]:
                        task["query_results"][query] = results
                        successfully_received.append(query)
                    else:
                        successfully_received.append(query)
                except Exception as e:
                    logger.error(f"处理查询 '{query}' 时出错: {e}")
                    failed_queries.append(query)
            
            task["last_activity"] = time.time()
            
            # 检查是否所有查询都已完成
            pending_queries = [q for q in task["queries"] if q not in task["query_results"]]
            
            all_completed = len(pending_queries) == 0
            
            # 改进的验证启动逻辑
            verification_started = False
            if all_completed:
                current_status = task["status"]
                
                # 检查是否需要启动验证
                if current_status == "waiting_for_queries":
                    print(f"[{datetime.now()}] 任务 {task_id} 所有查询已完成，准备启动验证")
                    logger.info(f"任务 {task_id} 所有查询已完成，准备启动验证")
                    task["status"] = "verifying"
                    
                    # 使用线程池提交验证任务
                    print(f"[{datetime.now()}] 提交验证任务到线程池: {task_id}")
                    future = task_executor.submit(verify_results, task_id, False)
                    
                    # 记录Future对象
                    with futures_lock:
                        active_futures[f"verify_{task_id}"] = future
                    
                    # 添加完成回调
                    def on_verify_complete(fut):
                        try:
                            fut.result()
                            print(f"[{datetime.now()}] 验证任务完成: {task_id}")
                            logger.info(f"验证任务完成: {task_id}")
                        except Exception as e:
                            print(f"[{datetime.now()}] 验证任务失败: {task_id}, 错误: {str(e)}")
                            logger.error(f"验证任务失败: {task_id}, 错误: {str(e)}")
                        finally:
                            with futures_lock:
                                active_futures.pop(f"verify_{task_id}", None)
                    
                    future.add_done_callback(on_verify_complete)
                    
                    verification_started = True
                    print(f"[{datetime.now()}] 验证任务已提交到线程池")
                    
                elif current_status == "verifying":
                    print(f"任务 {task_id} 已在验证中，跳过")
                elif current_status == "completed":
                    print(f"任务 {task_id} 已完成，跳过")
                else:
                    print(f"任务 {task_id} 状态异常: {current_status}")
        
        # 返回详细的确认信息
        return jsonify({
            "status": "success",
            "task_id": task_id,
            "successfully_received": successfully_received,
            "failed_queries": failed_queries,
            "total_received": len(successfully_received),
            "pending_queries": len(pending_queries),
            "total_queries": len(task["queries"]),
            "all_completed": all_completed,
            "verification_started": verification_started,
            "current_task_status": task["status"],
            "message": "Query results received with confirmation"
        })
        
    except Exception as e:
        logger.error(f"提交查询结果时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/get_received_queries/<task_id>', methods=['GET'])
def get_received_queries(task_id):
    """获取服务器已成功接收的查询列表"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        received_queries = list(task["query_results"].keys())
        
        return jsonify({
            "status": "success",
            "task_id": task_id,
            "received_queries": received_queries,
            "total_received": len(received_queries)
        })

@app.route('/client_exit/<task_id>', methods=['POST'])
def client_exit(task_id):
    """处理客户端退出通知"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        task["client_connected"] = False
        task["last_activity"] = time.time()
        
        print(f"客户端已断开连接，任务ID: {task_id}")
        logger.info(f"客户端已断开连接，任务ID: {task_id}")
        
        # 如果任务正在等待查询结果，则使用现有证据进行验证
        if task["status"] == "waiting_for_queries" and task["query_results"]:
            print(f"任务 {task_id} 使用现有证据进行验证")
            task["status"] = "verifying_partial"
            
            # 使用线程池提交验证任务
            future = task_executor.submit(verify_results, task_id, True)
            
            # 记录Future对象
            with futures_lock:
                active_futures[f"verify_partial_{task_id}"] = future
            
            return jsonify({
                "status": "success",
                "message": "Will proceed with partial evidence verification"
            })
        
        return jsonify({
            "status": "success", 
            "message": "Client disconnection recorded"
        })

@app.route('/get_verification_status/<task_id>', methods=['GET'])
def get_verification_status(task_id):
    """获取详细的验证状态"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        
        return jsonify({
            "task_id": task_id,
            "status": task["status"],
            "verification_stage": task.get("verification_stage", "unknown"),
            "verification_message": task.get("verification_message", ""),
            "last_activity": task["last_activity"],
            "time_since_last_activity": time.time() - task["last_activity"]
        })

@app.route('/get_task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        
        # 更新最后活动时间（如果客户端仍然连接）
        if task["client_connected"]:
            task["last_activity"] = time.time()
        
        response = {
            "status": task["status"],
            "task_id": task_id,
            "claim": task["claim"],
            "created_at": task["created_at"],
            "complete": task["complete"],
            "client_connected": task["client_connected"]
        }
        
        if task["status"] == "error":
            response["error"] = task.get("error", "Unknown error")
            response["traceback"] = task.get("traceback", "")
        
        if task["complete"]:
            response["result"] = task["result"]
        
        return jsonify(response)

@app.route('/download_result/<task_id>', methods=['GET'])
def download_result(task_id):
    """下载验证结果的JSON文件"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        
        if task["status"] != "completed" or "result_path" not in task:
            return jsonify({"error": "Result not available"}), 404
        
        result_path = task["result_path"]
    
    return send_file(result_path, as_attachment=True)

@app.route('/direct_verify/<task_id>', methods=['GET'])
def direct_verify(task_id):
    """使用Qwen直接验证(不需要外部查询)"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = dict(tasks[task_id])  # 创建副本
        
        if not task["media_path"]:
            return jsonify({"error": "No media file available for direct verification"}), 400
        
        # 更新最后活动时间
        tasks[task_id]["last_activity"] = time.time()
    
    try:
        print(f"开始直接验证任务 {task_id}...")
        logger.info(f"开始直接验证任务 {task_id}")
        
        # 创建新QwenModel实例并在完成后清理
        verification_result = create_and_use_qwen_model(
            task["claim"], 
            task["media_path"], 
            task["media_type"],
            enable_deepfake_detection=True
        )
        
        # 保存结果
        result_path = os.path.join(RESULTS_FOLDER, f"{task_id}_direct.json")
        
        result = {
            "task_id": task_id,
            "claim": task["claim"],
            "media_type": task["media_type"],
            "direct_verification": verification_result,
            "completed_at": time.time()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 更新任务状态
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["result_path"] = result_path
                tasks[task_id]["complete"] = True
                tasks[task_id]["result"] = result
                tasks[task_id]["last_activity"] = time.time()
        
        print(f"任务 {task_id} 直接验证完成。")
        logger.info(f"任务 {task_id} 直接验证完成")
        
        return jsonify({
            "status": "success",
            "task_id": task_id,
            "direct_verification": verification_result
        })
        
    except Exception as e:
        print(f"直接验证任务 {task_id} 时出错: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"直接验证任务 {task_id} 时出错: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def cleanup_inactive_tasks():
    """清理不活跃的任务"""
    while True:
        try:
            current_time = time.time()
            inactive_threshold = 1800  # 30分钟无活动则认为不活跃
            
            with tasks_lock:
                inactive_tasks = []
                for task_id, task in tasks.items():
                    if (current_time - task["last_activity"] > inactive_threshold and 
                        not task["complete"]):
                        inactive_tasks.append(task_id)
                
                for task_id in inactive_tasks:
                    print(f"清理不活跃任务: {task_id}")
                    logger.info(f"清理不活跃任务: {task_id}")
                    tasks[task_id]["client_connected"] = False
                    
                    # 如果任务有部分结果，尝试使用部分证据验证
                    if (tasks[task_id]["status"] == "waiting_for_queries" and 
                        tasks[task_id]["query_results"]):
                        print(f"任务 {task_id} 使用部分证据进行最终验证")
                        tasks[task_id]["status"] = "verifying_partial"
                        
                        # 使用线程池提交验证任务
                        future = task_executor.submit(verify_results, task_id, True)
                        
                        # 记录Future对象
                        with futures_lock:
                            active_futures[f"cleanup_verify_{task_id}"] = future
            
            time.sleep(300)  # 每5分钟检查一次
            
        except Exception as e:
            print(f"清理任务时出错: {e}")
            logger.error(f"清理任务时出错: {e}")
            time.sleep(300)

# 添加线程池状态监控端点
@app.route('/thread_pool_status', methods=['GET'])
def get_thread_pool_status():
    """获取线程池状态信息"""
    with futures_lock:
        active_tasks = list(active_futures.keys())
    
    return jsonify({
        "thread_pool_workers": task_executor._max_workers,
        "active_tasks": active_tasks,
        "active_task_count": len(active_tasks),
        "thread_name_prefix": "fact_check_"
    })

# 添加GPU内存状态监控端点
@app.route('/gpu_memory_status', methods=['GET'])
def get_gpu_memory_status():
    """获取GPU内存使用状态"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return jsonify({
                "gpu_available": True,
                "allocated_memory_gb": round(allocated, 2),
                "cached_memory_gb": round(cached, 2),
                "max_allocated_gb": round(max_allocated, 2),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device()
            })
        else:
            return jsonify({
                "gpu_available": False,
                "message": "No GPU available"
            })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "gpu_available": False
        })

# 添加手动清理内存的端点
@app.route('/force_cleanup_memory', methods=['POST'])
def force_cleanup_memory_endpoint():
    """手动触发GPU内存清理"""
    try:
        force_cleanup_gpu_memory()
        return jsonify({
            "status": "success",
            "message": "GPU memory cleanup completed"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        })

if __name__ == '__main__':
    print("事实核查服务器正在启动...")
    print("配置信息:")
    print(f"Deepseek模型路径: {config.DEEPSEEK_MODEL_PATH}")
    print(f"Qwen模型路径: {config.QWEN_MODEL_PATH}")
    print(f"监听地址: 0.0.0.0:8081")
    print(f"线程池工作线程数: {task_executor._max_workers}")
    print("已启用推理后内存自动清理模式")
    
    logger.info("事实核查服务器正在启动...")
    logger.info(f"线程池工作线程数: {task_executor._max_workers}")
    logger.info("已启用推理后内存自动清理模式")
    
    # 启动清理线程
    cleanup_thread = threading.Thread(target=cleanup_inactive_tasks, name="cleanup_thread")
    cleanup_thread.daemon = True
    cleanup_thread.start()
    print("任务清理线程已启动")
    logger.info("任务清理线程已启动")
    
    # 使用werkzeug的run_simple替代app.run可以获得更多控制
    run_simple('0.0.0.0', 8081, app, use_reloader=False, use_debugger=True, 
               threaded=True, processes=1, ssl_context=None, 
               passthrough_errors=False)