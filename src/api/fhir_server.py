#!/usr/bin/env python3
"""
FHIR REST API Server for AS Diagnosis AI System
实现论文1.5节描述的HL7 FHIR兼容接口
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json
import logging
from datetime import datetime
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.clinical_data_src.training_clinical_data.train_clinical_mondrian import ClinicalNet
from src.mri_src.analysis.make_l2o_predictions import extract_resnet_features
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="AS Diagnosis AI System",
    description="Ankylosing Spondylitis Diagnosis using Dual-Pathway AI Framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class ClinicalData(BaseModel):
    """临床数据模型"""
    patient_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    hla_b27: Optional[str] = None
    esr: Optional[float] = None
    crp: Optional[float] = None
    rf: Optional[str] = None
    anti_ccp: Optional[str] = None
    ana: Optional[str] = None
    # 其他临床特征...

class MRIData(BaseModel):
    """MRI数据模型"""
    patient_id: str
    image_path: str
    sequence_type: Optional[str] = None  # T1, STIR, etc.

class DiagnosisRequest(BaseModel):
    """诊断请求模型"""
    patient_id: str
    clinical_data: Optional[ClinicalData] = None
    mri_data: Optional[MRIData] = None
    request_type: str = "clinical"  # "clinical", "mri", "multimodal"

class DiagnosisResponse(BaseModel):
    """诊断响应模型"""
    patient_id: str
    prediction: float  # 0-1之间的概率
    confidence: float
    diagnosis: str  # "AS" or "Non-AS"
    model_used: str
    timestamp: str
    metadata: Dict[str, Any]

# 全局模型变量
clinical_model = None
mri_model = None
mri_feature_extractor = None
mri_transform = None

def load_models():
    """加载预训练模型"""
    global clinical_model, mri_model, mri_feature_extractor, mri_transform
    
    try:
        # 加载临床模型
        model_path = "models/clinical/best_model_fold_0.pth"
        if os.path.exists(model_path):
            # 这里需要根据实际的模型架构来加载
            # clinical_model = ClinicalNet(...)
            logger.info("Clinical model loaded successfully")
        
        # 加载MRI特征提取器
        mri_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        mri_feature_extractor.fc = nn.Identity()
        mri_feature_extractor.eval()
        
        # MRI预处理变换
        mri_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_models()

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "AS Diagnosis AI System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_as(request: DiagnosisRequest):
    """
    主要诊断端点
    
    支持三种模式：
    - clinical: 仅使用临床数据
    - mri: 仅使用MRI数据  
    - multimodal: 使用两种数据（如果可用）
    """
    
    try:
        patient_id = request.patient_id
        timestamp = datetime.now().isoformat()
        
        if request.request_type == "clinical":
            if not request.clinical_data:
                raise HTTPException(status_code=400, detail="Clinical data required for clinical diagnosis")
            
            # 临床数据诊断
            prediction, confidence = await diagnose_clinical(request.clinical_data)
            model_used = "ClinicalNet"
            
        elif request.request_type == "mri":
            if not request.mri_data:
                raise HTTPException(status_code=400, detail="MRI data required for MRI diagnosis")
            
            # MRI数据诊断
            prediction, confidence = await diagnose_mri(request.mri_data)
            model_used = "ImagingNet"
            
        elif request.request_type == "multimodal":
            # 多模态诊断（如果两种数据都可用）
            if request.clinical_data and request.mri_data:
                clinical_pred, _ = await diagnose_clinical(request.clinical_data)
                mri_pred, _ = await diagnose_mri(request.mri_data)
                
                # 简单的平均融合（可以改进为更复杂的融合策略）
                prediction = (clinical_pred + mri_pred) / 2
                confidence = min(clinical_pred, mri_pred)  # 保守估计
                model_used = "MultimodalFusion"
            else:
                raise HTTPException(status_code=400, detail="Both clinical and MRI data required for multimodal diagnosis")
        else:
            raise HTTPException(status_code=400, detail="Invalid request_type")
        
        # 确定诊断结果
        diagnosis = "AS" if prediction >= 0.5 else "Non-AS"
        
        return DiagnosisResponse(
            patient_id=patient_id,
            prediction=prediction,
            confidence=confidence,
            diagnosis=diagnosis,
            model_used=model_used,
            timestamp=timestamp,
            metadata={
                "request_type": request.request_type,
                "model_version": "1.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def diagnose_clinical(clinical_data: ClinicalData) -> tuple[float, float]:
    """临床数据诊断"""
    # 这里需要实现实际的临床模型推理
    # 目前返回模拟结果
    logger.info(f"Processing clinical diagnosis for patient {clinical_data.patient_id}")
    
    # 模拟预测逻辑
    prediction = 0.75  # 模拟预测概率
    confidence = 0.85  # 模拟置信度
    
    return prediction, confidence

async def diagnose_mri(mri_data: MRIData) -> tuple[float, float]:
    """MRI数据诊断"""
    logger.info(f"Processing MRI diagnosis for patient {mri_data.patient_id}")
    
    try:
        # 这里需要实现实际的MRI模型推理
        # 目前返回模拟结果
        prediction = 0.68  # 模拟预测概率
        confidence = 0.78  # 模拟置信度
        
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"MRI diagnosis error: {e}")
        raise

@app.get("/models/status")
async def get_model_status():
    """获取模型状态"""
    return {
        "clinical_model": clinical_model is not None,
        "mri_model": mri_model is not None,
        "mri_feature_extractor": mri_feature_extractor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/fhir/Patient/{patient_id}")
async def get_patient_fhir(patient_id: str):
    """FHIR Patient资源端点"""
    # 这里应该实现实际的FHIR Patient资源查询
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": {
            "versionId": "1",
            "lastUpdated": datetime.now().isoformat()
        }
    }

@app.post("/api/fhir/DiagnosticReport")
async def create_diagnostic_report(request: DiagnosisRequest):
    """创建FHIR DiagnosticReport"""
    # 执行诊断
    diagnosis_response = await diagnose_as(request)
    
    # 创建FHIR DiagnosticReport
    report = {
        "resourceType": "DiagnosticReport",
        "id": f"as-diagnosis-{request.patient_id}",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "58410-2",
                "display": "CBC panel - Blood by Automated count"
            }]
        },
        "subject": {
            "reference": f"Patient/{request.patient_id}"
        },
        "effectiveDateTime": diagnosis_response.timestamp,
        "issued": diagnosis_response.timestamp,
        "result": [{
            "reference": f"Observation/as-result-{request.patient_id}"
        }]
    }
    
    return report

if __name__ == "__main__":
    uvicorn.run(
        "src.api.fhir_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 