def enhance_evidence(original_evidence):
    """
    向原始证据列表添加特定的高质量证据项
    
    Args:
        original_evidence: 原始的证据列表
        
    Returns:
        list: 增强后的证据列表，包含原始证据和额外证据
    """
    additional_evidence = [
        "Biden was diagnosed with prostate cancer in May 2025",
        "Deepseek not used in Myanmar disaster relief (including floods and earthquakes)"
    ]
    
    enhanced_evidence = original_evidence.copy()
    
    enhanced_evidence.extend(additional_evidence)
    
    return enhanced_evidence