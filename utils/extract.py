import os
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis
from utils.retrieval_utils import FaissRetriever

class FaceDetector:
    """鲁棒人脸检测器，使用 InsightFace 提取特征"""
    def __init__(self):
        self.detector = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.detector.prepare(ctx_id=0, det_size=(512, 512))

    def get_face_feature(self, img_path, max_attempts=3):
        """从图像中提取人脸特征，失败时重试多次"""
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            return None, True  # 返回 None 和失败标志
        
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 尝试多次检测
        for attempt in range(max_attempts):
            faces = self.detector.get(img)
            if faces and len(faces) > 0:
                return faces[0].normed_embedding, False  # 返回特征和成功标志
            else:
                print(f"Attempt {attempt + 1}/{max_attempts}: No face detected in {img_path}")
        
        # 所有尝试失败后返回 None
        print(f"All {max_attempts} attempts failed for {img_path}")
        return None, True  # 返回 None 和失败标志

class FaceIndexBuilder:
    """从低质图片数据集中提取人脸特征的工具类，仅保存有效特征并构建 Faiss 索引"""
    
    def __init__(self, img_dir: str, data_json_path: str, output_dir: str):
        """
        初始化构建器
        
        Args:
            img_dir: 图片数据集目录
            data_json_path: data_json 文件路径
            output_dir: 输出目录
        """
        self.img_dir = img_dir
        self.data_json_path = data_json_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        self.feature_dir = os.path.join(output_dir)
        os.makedirs(self.feature_dir, exist_ok=True)

        # 加载 data_json 并检查格式
        try:
            with open(data_json_path, 'r') as f:
                self.data_json = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {data_json_path}: {str(e)}")
        
        # 统一使用 "image_file" 作为键名
        self.image_files = []
        for item in self.data_json:
            # 如果 JSON 中是 "low_quality_file"，重命名为 "image_file"
            image_file = item.get("low_quality_file", item.get("image_file"))
            if image_file:
                self.image_files.append(image_file)
            else:
                print(f"Warning: No valid image file key in JSON item: {item}")
        print(f"从 data_json 中找到 {len(self.image_files)} 个图片")

        self.face_detector = FaceDetector()

    def extract_features(self) -> tuple[np.ndarray, list]:
        """从图片中提取人脸特征，仅保存有效特征，并记录失败图片"""
        features = []
        valid_indices = []
        failed_images = []  # 记录检测失败的图片

        for idx, image_file in enumerate(tqdm(self.image_files, desc="提取图片特征")):
            img_path = os.path.join(self.img_dir, image_file)
            feature, failed = self.face_detector.get_face_feature(img_path)
            if feature is not None:
                features.append(feature)
                valid_indices.append(idx)
            else:
                failed_images.append(image_file)

        if not features:
            raise ValueError("未提取到任何有效人脸特征")
        
        feature_array = np.stack(features)
        features_path = os.path.join(self.feature_dir, "all_features.npy")
        np.save(features_path, feature_array)
        print(f"特征矩阵保存至: {features_path}, 形状: {feature_array.shape}")

        # 保存检测失败的图片到 JSON
        if failed_images:
            failed_path = os.path.join(self.feature_dir, "failed_images.json")
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(failed_images, f, ensure_ascii=False, indent=2)
            print(f"检测失败的 {len(failed_images)} 张图片保存至: {failed_path}")

        return feature_array, valid_indices

    def generate_key_mapping(self, valid_indices: list) -> dict:
        """生成键映射并保存为 face_key_mapping.json，仅包含有效图片"""
        key_mapping = {self.image_files[idx]: i for i, idx in enumerate(valid_indices)}
        
        key_mapping_path = os.path.join(self.feature_dir, "face_key_mapping.json")
        with open(key_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(key_mapping, f, ensure_ascii=False, indent=2)
        print(f"键映射保存至: {key_mapping_path}, 有效图片数: {len(key_mapping)}")

        return key_mapping

    def build_faiss_index(self, features: np.ndarray, key_mapping: dict) -> None:
        """构建并保存 Faiss 索引"""
        # 创建检索器实例，使用余弦相似度（"ip"）或 L2 距离（"l2"）
        retriever = FaissRetriever(
            index_type="ip",  # 默认使用余弦相似度
            nlist=100,        # IVF 索引的簇数
            nprobe=10,        # 搜索时探测的簇数
            top_k=5           # 返回的结果数量
        )
        
        # 构建索引
        retriever.build_index(
            features=features,
            key_mapping=key_mapping
        )

        # 保存索引和相关文件
        index_path = os.path.join(self.feature_dir, "face_index.faiss")
        features_path = os.path.join(self.feature_dir, "all_features.npy")
        key_mapping_path = os.path.join(self.feature_dir, "face_key_mapping.json")
        
        retriever.save(
            index_path=index_path,
            features_path=features_path,
            key_mapping_path=key_mapping_path
        )
        print(f"Faiss 索引保存至: {index_path}")

if __name__ == "__main__":
    # 配置路径
    img_dir = "/data/IP-retrieval/dataset"
    data_json_path = "/data/IP-retrieval/dataset/data.json"
    output_dir = "/data/IP-retrieval/dataset/feature/low"

    # 初始化 FaceIndexBuilder
    builder = FaceIndexBuilder(
        img_dir=img_dir,
        data_json_path=data_json_path,
        output_dir=output_dir
    )

    # 提取人脸特征
    features, valid_indices = builder.extract_features()

    # 生成键映射
    key_mapping = builder.generate_key_mapping(valid_indices)

    # 构建 Faiss 索引
    builder.build_faiss_index(features, key_mapping)

    # 输出结果
    key_mapping_path = os.path.join(builder.feature_dir, "face_key_mapping.json")
    features_path = os.path.join(builder.feature_dir, "all_features.npy")
    index_path = os.path.join(builder.feature_dir, "face_index.faiss")
    failed_path = os.path.join(builder.feature_dir, "failed_images.json")
    
    print(f"完成！")
    print(f"- 特征矩阵: {features_path}")
    print(f"- 键映射: {key_mapping_path}")
    print(f"- Faiss 索引: {index_path}")
    if os.path.exists(failed_path):
        print(f"- 检测失败图片: {failed_path}")