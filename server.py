from flask import Flask, request, jsonify, send_file
import os
import json
import uuid
import tempfile
import time
import shutil
from werkzeug.utils import secure_filename
import traceback
from processors.claim_processor import ClaimProcessor
from processors.video_processor import VideoProcessor
from processors.image_processor import ImageProcessor
from fact_checker.verifier import FactVerifier
from models.qwen_model import QwenModel
import config
from werkzeug.serving import run_simple
import threading

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

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/submit_task', methods=['POST'])
def submit_task():
    """接收媒体文件和声明，启动分析任务"""
    try:
        print("收到任务提交请求...")
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
        
        # 启动处理（为了避免请求超时，我们把处理放到后台）
        thread = threading.Thread(target=process_task, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success", 
            "task_id": task_id,
            "message": "Task submitted successfully"
        })
        
    except Exception as e:
        print(f"提交任务时出错: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def process_task(task_id):
    """处理任务，提取需要查询的内容"""
    print(f"开始处理任务 {task_id}...")
    
    with tasks_lock:
        if task_id not in tasks:
            print(f"任务 {task_id} 不存在")
            return
        task = dict(tasks[task_id])  # 创建副本以避免长时间持有锁
    
    try:
        # 处理声明
        print(f"处理声明: {task['claim']}")
        claim_processor = ClaimProcessor(model_path=config.DEEPSEEK_MODEL_PATH)
        claim_sentences = claim_processor.process(task['claim'])
        
        # 更新任务状态
        with tasks_lock:
            if task_id in tasks and tasks[task_id]["client_connected"]:
                tasks[task_id]["claim_sentences"] = claim_sentences
                tasks[task_id]["queries"] = claim_sentences.copy()  # 初始化查询列表
                tasks[task_id]["last_activity"] = time.time()
        
        # 处理媒体文件（如果有）
        if task["media_path"] and task["media_type"]:
            if task["media_type"] == 'video':
                # 处理视频
                print(f"处理视频: {task['media_path']}")
                video_processor = VideoProcessor(
                    model_path=config.QWEN_MODEL_PATH,
                    cache_dir=config.MODEL_CACHE_DIR
                )
                video_content = video_processor.process(task['media_path'])
                
                # 处理文本描述
                visual_sentences = claim_processor.process(video_content["visual_content"])
                audio_sentences = claim_processor.process(video_content["audio_content"])
                
                # 更新任务状态
                with tasks_lock:
                    if task_id in tasks and tasks[task_id]["client_connected"]:
                        tasks[task_id]["visual_content"] = video_content["visual_content"]
                        tasks[task_id]["audio_content"] = video_content["audio_content"]
                        tasks[task_id]["media_content"] = video_content["full_content"]
                        tasks[task_id]["visual_sentences"] = visual_sentences
                        tasks[task_id]["audio_sentences"] = audio_sentences
                        tasks[task_id]["queries"].extend(visual_sentences)
                        tasks[task_id]["queries"].extend(audio_sentences)
                        tasks[task_id]["last_activity"] = time.time()
                
            elif task["media_type"] == 'image':
                # 处理图像
                print(f"处理图像: {task['media_path']}")
                image_processor = ImageProcessor(
                    model_path=config.QWEN_MODEL_PATH,
                    cache_dir=config.MODEL_CACHE_DIR
                )
                image_text = image_processor.process(task['media_path'])
                
                # 处理图像描述
                image_sentences = claim_processor.process(image_text)
                
                # 更新任务状态
                with tasks_lock:
                    if task_id in tasks and tasks[task_id]["client_connected"]:
                        tasks[task_id]["media_content"] = image_text
                        tasks[task_id]["image_sentences"] = image_sentences
                        tasks[task_id]["queries"].extend(image_sentences)
                        tasks[task_id]["last_activity"] = time.time()
        
        # 更新任务状态为等待查询
        with tasks_lock:
            if task_id in tasks and tasks[task_id]["client_connected"]:
                tasks[task_id]["status"] = "waiting_for_queries"
                tasks[task_id]["last_activity"] = time.time()
                print(f"任务 {task_id} 处理完成。{len(tasks[task_id]['queries'])} 个查询准备就绪。")
            else:
                print(f"任务 {task_id} 客户端已断开连接，停止处理")
                return
        
    except Exception as e:
        print(f"处理任务 {task_id} 时出错: {str(e)}")
        print(traceback.format_exc())
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
    """接收客户端的查询结果"""
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
        
        # 更新任务的查询结果
        with tasks_lock:
            task["query_results"].update(query_results)
            task["last_activity"] = time.time()
            
            # 检查是否所有查询都已完成
            pending_queries = [q for q in task["queries"] if q not in task["query_results"]]
            
            if not pending_queries:
                # 所有查询已完成，开始验证
                task["status"] = "verifying"
                # 异步执行结果验证
                thread = threading.Thread(target=verify_results, args=(task_id,))
                thread.daemon = True
                thread.start()
        
        return jsonify({
            "status": "success",
            "task_id": task_id,
            "pending_queries": len(pending_queries),
            "total_queries": len(task["queries"]),
            "message": "Query results received"
        })
        
    except Exception as e:
        print(f"提交查询结果时出错: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

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
        
        # 如果任务正在等待查询结果，则使用现有证据进行验证
        if task["status"] == "waiting_for_queries" and task["query_results"]:
            print(f"任务 {task_id} 使用现有证据进行验证")
            task["status"] = "verifying_partial"
            
            # 异步执行结果验证（使用部分证据）
            thread = threading.Thread(target=verify_results, args=(task_id, True))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "status": "success",
                "message": "Will proceed with partial evidence verification"
            })
        
        return jsonify({
            "status": "success", 
            "message": "Client disconnection recorded"
        })

def verify_results(task_id, use_partial_evidence=False):
    """使用查询结果进行事实验证"""
    print(f"开始验证任务 {task_id}... (部分证据: {use_partial_evidence})")
    
    with tasks_lock:
        if task_id not in tasks:
            print(f"任务 {task_id} 不存在")
            return
        task = dict(tasks[task_id])  # 创建副本
    
    try:
        # 准备证据
        all_evidence = []
        for query, results in task["query_results"].items():
            if isinstance(results, list):
                all_evidence.extend(results)
        
        # 去重
        all_evidence = list(set(all_evidence))
        evidence_count = len(all_evidence)
        
        if use_partial_evidence:
            print(f"使用部分证据进行验证，收集到 {evidence_count} 条证据")
        else:
            print(f"使用完整证据进行验证，收集到 {evidence_count} 条证据")
        
        # 进行事实验证
        verifier = FactVerifier(model_path=config.DEEPSEEK_MODEL_PATH)
        
        # 初步验证
        print("执行初步验证...")
        initial_judgment = verifier.initial_verify(
            task["claim"], 
            task.get("media_content")
        )
        
        # 基于证据的最终验证（即使证据较少也进行验证）
        if evidence_count > 0:
            print("基于证据执行最终验证...")
            final_judgment = verifier.verify_with_evidence(
                task["claim"], 
                task.get("media_content"), 
                all_evidence
            )
        else:
            print("没有可用证据，使用初步验证结果")
            final_judgment = {
                "final_judgment": initial_judgment.get("initial_judgment", "uncertain"),
                "confidence": max(0, initial_judgment.get("confidence", 0.5) - 0.1),  # 降低置信度
                "reasoning": f"基于媒体内容的判断，未获得外部证据支持。{initial_judgment.get('reasoning', '')}",
                "evidence_analysis": []
            }
        
        # 保存结果
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
        
        if use_partial_evidence:
            print(f"任务 {task_id} 部分证据验证完成。")
        else:
            print(f"任务 {task_id} 完整验证完成。")
        
    except Exception as e:
        print(f"验证任务 {task_id} 时出错: {str(e)}")
        print(traceback.format_exc())
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = str(e)
                tasks[task_id]["traceback"] = traceback.format_exc()
                tasks[task_id]["last_activity"] = time.time()

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
        qwen_model = QwenModel(
            model_path=config.QWEN_MODEL_PATH,
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        verification_result = qwen_model.verify_claim(
            task["claim"], 
            task["media_path"], 
            task["media_type"]
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
        
        return jsonify({
            "status": "success",
            "task_id": task_id,
            "direct_verification": verification_result
        })
        
    except Exception as e:
        print(f"直接验证任务 {task_id} 时出错: {str(e)}")
        print(traceback.format_exc())
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
                    tasks[task_id]["client_connected"] = False
                    
                    # 如果任务有部分结果，尝试使用部分证据验证
                    if (tasks[task_id]["status"] == "waiting_for_queries" and 
                        tasks[task_id]["query_results"]):
                        print(f"任务 {task_id} 使用部分证据进行最终验证")
                        tasks[task_id]["status"] = "verifying_partial"
                        
                        # 异步执行验证
                        thread = threading.Thread(target=verify_results, args=(task_id, True))
                        thread.daemon = True
                        thread.start()
            
            time.sleep(300)  # 每5分钟检查一次
            
        except Exception as e:
            print(f"清理任务时出错: {e}")
            time.sleep(300)

if __name__ == '__main__':
    print("事实核查服务器正在启动...")
    print("配置信息:")
    print(f"Deepseek模型路径: {config.DEEPSEEK_MODEL_PATH}")
    print(f"Qwen模型路径: {config.QWEN_MODEL_PATH}")
    print(f"监听地址: 0.0.0.0:8081")
    
    # 启动清理线程
    cleanup_thread = threading.Thread(target=cleanup_inactive_tasks)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    print("任务清理线程已启动")
    
    # 使用werkzeug的run_simple替代app.run可以获得更多控制
    run_simple('0.0.0.0', 8081, app, use_reloader=False, use_debugger=True, 
               threaded=True, processes=1, ssl_context=None, 
               passthrough_errors=False)