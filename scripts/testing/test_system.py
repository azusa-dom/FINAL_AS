#!/usr/bin/env python3
"""
AS诊断AI系统测试脚本
验证各个组件的功能
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class TestClinicalPipeline(unittest.TestCase):
    """测试临床数据管道"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # 创建测试数据
        self.create_test_clinical_data()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_clinical_data(self):
        """创建测试临床数据"""
        # 创建模拟的临床数据
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Patient_ID': [f'P{i:03d}' for i in range(n_samples)],
            'Age': np.random.normal(40, 10, n_samples),
            'Sex': np.random.choice(['M', 'F'], n_samples),
            'HLA_B27': np.random.choice(['positive', 'negative'], n_samples),
            'ESR': np.random.normal(30, 15, n_samples),
            'CRP': np.random.normal(15, 8, n_samples),
            'RF': np.random.choice(['positive', 'negative'], n_samples),
            'Anti_CCP': np.random.choice(['positive', 'negative'], n_samples),
            'ANA': np.random.choice(['positive', 'negative'], n_samples),
            'Disease': np.random.choice(['Ankylosing Spondylitis', 'Rheumatoid Arthritis', 'Healthy'], n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.test_data_dir / "clinical_data.csv", index=False)
    
    def test_clinical_dataset(self):
        """测试临床数据集类"""
        from src.dataset import ClinicalDataset
        
        csv_path = self.test_data_dir / "clinical_data.csv"
        dataset = ClinicalDataset(str(csv_path), label_column="Disease", id_column="Patient_ID")
        
        self.assertEqual(len(dataset), 100)
        self.assertIsInstance(dataset.features, torch.Tensor)
        self.assertIsInstance(dataset.labels, torch.Tensor)
    
    def test_clinical_preprocessing(self):
        """测试临床数据预处理"""
        from src.clinical_data_src.clinical_data_preparation.preprocess_clinical_final import run_final_preprocessing
        
        input_csv = self.test_data_dir / "clinical_data.csv"
        output_dir = self.test_data_dir / "processed"
        
        run_final_preprocessing(str(input_csv), str(output_dir))
        
        # 检查输出文件
        self.assertTrue((output_dir / "fold_0_train.csv").exists())
        self.assertTrue((output_dir / "fold_0_val.csv").exists())

class TestMRIPipeline(unittest.TestCase):
    """测试MRI数据管道"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # 创建测试数据
        self.create_test_mri_data()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_mri_data(self):
        """创建测试MRI数据"""
        # 创建模拟的MRI图像目录结构
        mri_dir = self.test_data_dir / "mri"
        mri_dir.mkdir()
        
        # 创建AS患者目录
        as_dir = mri_dir / "mri_AS"
        as_dir.mkdir()
        
        # 创建健康对照目录
        health_dir = mri_dir / "mri_health"
        health_dir.mkdir()
        
        # 创建模拟图像文件
        for i in range(3):
            # AS患者
            patient_dir = as_dir / f"patient_{i}"
            patient_dir.mkdir()
            
            # 创建模拟图像文件
            for j in range(5):
                img_path = patient_dir / f"slice_{j}.png"
                # 创建简单的测试图像
                from PIL import Image
                img = Image.new('RGB', (224, 224), color=(i * 50, j * 50, 100))
                img.save(img_path)
        
        # 健康对照
        for i in range(2):
            patient_dir = health_dir / f"health_{i}"
            patient_dir.mkdir()
            
            for j in range(5):
                img_path = patient_dir / f"slice_{j}.png"
                from PIL import Image
                img = Image.new('RGB', (224, 224), color=(100, i * 50, j * 50))
                img.save(img_path)
    
    def test_mri_dataset(self):
        """测试MRI数据集类"""
        from src.dataset import MRIDataset
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        dataset = MRIDataset(str(self.test_data_dir / "mri"), transform=transform)
        
        self.assertGreater(len(dataset), 0)
        
        # 测试数据加载
        image, label, patient_id = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertIsInstance(patient_id, str)
    
    def test_direction_correction(self):
        """测试方向性校正"""
        from src.mri_src.analysis.mri_direction_correction import apply_direction_correction
        
        # 创建测试预测数据
        test_data = {
            'subject_id': ['P001', 'P002', 'P003', 'P004'],
            'y_true': [1, 1, 0, 0],
            'prob_raw': [0.3, 0.4, 0.7, 0.8],  # 故意创建低AUROC的情况
            'logit_raw': [-0.85, -0.41, 0.85, 1.39]
        }
        
        df = pd.DataFrame(test_data)
        corrected_df = apply_direction_correction(df)
        
        self.assertIn('prob_corrected', corrected_df.columns)
        self.assertIn('logit_corrected', corrected_df.columns)

class TestFeatureAnalysis(unittest.TestCase):
    """测试特征空间分析"""
    
    def test_distance_statistics(self):
        """测试距离统计计算"""
        from src.mri_src.mri_feature_analysis.feature_space_geometry import compute_distance_statistics
        
        # 创建测试特征数据
        np.random.seed(42)
        n_samples = 20
        n_features = 512
        
        # 创建两个类别的特征
        features = np.random.randn(n_samples, n_features)
        labels = np.array([0] * 10 + [1] * 10)  # 10个样本每个类别
        
        # 计算距离统计
        distance_stats = compute_distance_statistics(features, labels)
        
        self.assertIn('euclidean_distances', distance_stats)
        self.assertIn('cosine_distances', distance_stats)
        self.assertIn('ks_results', distance_stats)
    
    def test_embedding_projections(self):
        """测试嵌入投影"""
        from src.mri_src.mri_feature_analysis.feature_space_geometry import compute_embedding_projections
        
        # 创建测试特征数据
        np.random.seed(42)
        n_samples = 20
        n_features = 512
        
        features = np.random.randn(n_samples, n_features)
        labels = np.array([0] * 10 + [1] * 10)
        
        # 计算投影
        projection_results = compute_embedding_projections(features, labels, methods=['pca', 'kernel_pca'])
        
        self.assertIn('pca', projection_results)
        self.assertIn('kernel_pca', projection_results)
        self.assertIn('silhouette_score', projection_results['pca'])

class TestAPI(unittest.TestCase):
    """测试API功能"""
    
    def test_fhir_server_import(self):
        """测试FHIR服务器导入"""
        try:
            from src.api.fhir_server import app
            self.assertIsNotNone(app)
        except ImportError as e:
            self.fail(f"无法导入FHIR服务器: {e}")
    
    def test_api_models(self):
        """测试API数据模型"""
        from src.api.fhir_server import ClinicalData, MRIData, DiagnosisRequest
        
        # 测试临床数据模型
        clinical_data = ClinicalData(
            patient_id="P001",
            age=35.0,
            sex="M",
            hla_b27="positive",
            esr=45.2,
            crp=18.5
        )
        self.assertEqual(clinical_data.patient_id, "P001")
        
        # 测试MRI数据模型
        mri_data = MRIData(
            patient_id="P001",
            image_path="/path/to/image.nii.gz",
            sequence_type="T1"
        )
        self.assertEqual(mri_data.patient_id, "P001")
        
        # 测试诊断请求模型
        request = DiagnosisRequest(
            patient_id="P001",
            clinical_data=clinical_data,
            request_type="clinical"
        )
        self.assertEqual(request.patient_id, "P001")

class TestConfiguration(unittest.TestCase):
    """测试配置系统"""
    
    def test_config_import(self):
        """测试配置导入"""
        try:
            sys.path.append(project_root)
            from config import ClinicalConfig, MRIConfig, APIConfig
            self.assertIsNotNone(ClinicalConfig)
            self.assertIsNotNone(MRIConfig)
            self.assertIsNotNone(APIConfig)
        except ImportError as e:
            self.fail(f"无法导入配置: {e}")
    
    def test_config_values(self):
        """测试配置值"""
        sys.path.append(project_root)
        from config import ClinicalConfig, MRIConfig
        
        # 测试临床配置
        self.assertEqual(ClinicalConfig.HIDDEN_SIZE, 64)
        self.assertEqual(ClinicalConfig.LEARNING_RATE, 1e-3)
        self.assertEqual(ClinicalConfig.N_SPLITS, 5)
        
        # 测试MRI配置
        self.assertEqual(MRIConfig.TARGET_SHAPE, (224, 224))
        self.assertEqual(MRIConfig.FEATURE_DIM, 512)
        self.assertEqual(MRIConfig.BACKBONE, "resnet18")

def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestClinicalPipeline,
        TestMRIPipeline,
        TestFeatureAnalysis,
        TestAPI,
        TestConfiguration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()

if __name__ == "__main__":
    print("开始运行AS诊断AI系统测试...")
    
    success = run_all_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1) 