import os
import numpy as np
import torch
import logging
import faiss
from typing import List, Tuple, Dict, Optional, Union
import torch.nn.functional as F
import json
import time

logger = logging.getLogger(__name__)

def safe_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-6):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, np.inf)
    
class FaissRetriever:
    """
    用 Faiss 库实现的特征检索器，支持高效的人脸特征检索。
    修改：支持余弦相似度（IndexFlatIP + 单位归一化）或 L2 距离（IndexFlatL2）。
    通过 image_file 文件名映射到特征索引，依赖 all_features.npy 作为特征来源。
    """
    def __init__(
        self,
        index_path=None,
        features_path=None,
        key_mapping_path=None,
        index_type="ip",
        top_k_final=5,
        nlist=100,
        nprobe=10,
        top_k_rough=50,
        use_gpu=False
    ):
        """
        Args:
            index_path (str): Faiss 索引文件路径
            features_path (str): 特征数据文件路径 (all_features.npy)
            key_mapping_path (str): 映射文件路径，格式为 {"image_file": index}
            index_type (str): 索引类型，支持 "ip"（余弦相似度）, "l2"（L2 距离）
            top_k (int): 返回的最相似特征数量
            nlist (int): IVF 索引的簇数
            nprobe (int): IVF 索引的搜索簇数
            top_k_rough (int): 粗筛阶段返回的候选数量
            use_gpu (bool): 是否使用 GPU 加速
        """
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.top_k_rough = top_k_rough
        self.top_k_final = top_k_final
        self.use_gpu = use_gpu
        self.rough_index = None
        self.features = None
        self.key_mapping = None
        self.rev_key_mapping = None

        try:
            # 加载特征数据 (all_features.npy)
            if features_path and os.path.exists(features_path):
                self.features = np.load(features_path).astype(np.float32)
                if self.index_type == "ip":
                    # 单位归一化特征
                    self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
                logger.info(f"Loaded features from {features_path}, shape: {self.features.shape}")

            # 加载键映射，格式为 {"celeba-hq/train/000004.jpg": 0, ...}
            if key_mapping_path and os.path.exists(key_mapping_path):
                with open(key_mapping_path, 'r') as f:
                    self.key_mapping = json.load(f)
                self.rev_key_mapping = {v: k for k, v in self.key_mapping.items()}
                logger.info(f"Loaded key mapping from {key_mapping_path}, sample: {list(self.key_mapping.items())[:5]}")

            # 加载或构建索引
            if index_path and os.path.exists(index_path):
                self.rough_index = faiss.read_index(index_path)
                logger.info(f"Loaded Faiss index from {index_path}")
            elif self.features is not None:
                self._build_ivf_index()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _build_ivf_index(self):
        """构建 IVF 索引"""
        metric = faiss.METRIC_INNER_PRODUCT if self.index_type == "ip" else faiss.METRIC_L2
        quantizer = faiss.IndexFlatIP(self.features.shape[1]) if self.index_type == "ip" else faiss.IndexFlatL2(self.features.shape[1])
        self.rough_index = faiss.IndexIVFFlat(quantizer, self.features.shape[1], self.nlist, metric)
        self.rough_index.train(self.features)
        self.rough_index.add(self.features)
        self.rough_index.nprobe = self.nprobe

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.rough_index = faiss.index_cpu_to_gpu(res, 0, self.rough_index)
        logger.info(f"IVF index built with nlist={self.nlist}, nprobe={self.nprobe}, features={len(self.features)}")

    def build_index(self, features: Union[np.ndarray, torch.Tensor], key_mapping: Optional[Dict] = None):
        """
        构建索引
        Args:
            features: 特征矩阵，形状为 [N, D]，支持 np.ndarray 或 torch.Tensor
            key_mapping (Dict): image_file 到索引的映射，例如 {"celeba-hq/train/000004.jpg": 0}
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if features.dtype != np.float32:
            logger.info(f"Converting features to float32 (from {features.dtype})")
            features = features.astype(np.float32)

        if self.index_type == "ip":
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

        self.features = features
        if key_mapping:
            self.key_mapping = key_mapping
            self.rev_key_mapping = {v: k for k, v in key_mapping.items()}
        
        self._build_ivf_index()
        logger.info(f"Built {self.index_type} index with {features.shape[0]} features")

    def save(self, index_path: str, features_path: Optional[str] = None, key_mapping_path: Optional[str] = None):
        try:
            faiss.write_index(self.rough_index, index_path)
            if features_path and self.features is not None:
                np.save(features_path, self.features)
            if key_mapping_path and self.key_mapping is not None:
                with open(key_mapping_path, 'w') as f:
                    json.dump(self.key_mapping, f)
            logger.info(f"Saved index to {index_path}")
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise

    def search(self, query_features: Union[torch.Tensor, np.ndarray], k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        两阶段检索：
        1. 使用 IVF 索引进行粗筛
        2. 在粗筛结果上使用 FlatIP/FlatL2 进行精筛
        
        Args:
            query_features: 查询特征，形状为 [B, D]
            k: 返回的最相似特征数量
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - D: 相似度或距离，形状为 [B, K]
                - I: 索引位置，形状为 [B, K]
        """
        if self.rough_index is None:
            logger.error("Index not initialized")
            raise ValueError("Index not initialized")

        if isinstance(query_features, torch.Tensor):
            query_features = query_features.float().cpu().numpy()
        else:
            query_features = np.asarray(query_features, dtype=np.float32)

        if self.index_type == "ip":
            # query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
            query_features = safe_normalize(query_features)

        start_time = time.time()
        k = k or self.top_k_final
        B = query_features.shape[0]

        # 第一阶段：粗筛
        logger.debug(f"Performing rough search with top_k_rough={self.top_k_rough}")
        D_rough, I_rough = self.rough_index.search(query_features, self.top_k_rough)
        
        # 检查 I_rough 的有效性
        if I_rough.max() >= len(self.features):
            logger.error(f"I_rough contains invalid indices: max={I_rough.max()}, features size={len(self.features)}")
            raise ValueError("Invalid indices in I_rough")
        logger.debug(f"I_rough shape: {I_rough.shape}, max index: {I_rough.max()}")

        # 第二阶段：精筛（逐查询处理）
        final_D = np.zeros((B, k), dtype=np.float32)
        final_I = np.zeros((B, k), dtype=np.int64)

        for i in range(B):
            # 1. 获取当前查询的粗筛候选
            candidate_indices = I_rough[i]  # [top_k_rough]
            candidates = self.features[candidate_indices]  # [top_k_rough, D]
            
            # 2. 归一化（如果需要）
            if self.index_type == "ip":
                candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
                # query = query_features[i:i+1] / np.linalg.norm(query_features[i:i+1], axis=1, keepdims=True)
                query = safe_normalize(query_features[i:i+1])
            else:
                query = query_features[i:i+1]

            # 3. 构建精筛索引
            fine_index = faiss.IndexFlatIP(self.features.shape[1]) if self.index_type == "ip" else faiss.IndexFlatL2(self.features.shape[1])
            fine_index.add(candidates)  # 添加 top_k_rough 个候选

            # 4. 精筛检索
            D_fine, I_fine = fine_index.search(query, k)  # [1, k]
            logger.debug(f"Query {i}: I_fine={I_fine}, max={I_fine.max()}, top_k_rough={self.top_k_rough}")

            # 5. 检查 I_fine 的有效性
            if I_fine.max() >= self.top_k_rough:
                logger.error(f"I_fine contains invalid indices for query {i}: max={I_fine.max()}, top_k_rough={self.top_k_rough}")
                raise ValueError("Invalid indices in I_fine")

            # 6. 映射到全局索引
            final_D[i] = D_fine[0]
            final_I[i] = candidate_indices[I_fine[0]]

        logger.debug(f"Two-stage search completed in {time.time() - start_time:.4f}s for {B} queries")
        return final_D, final_I
    
    def get_vectors(self, indices: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """获取给定索引的特征向量"""
        if self.features is None:
            logger.warning("Features not initialized")
            return None

        indices = torch.as_tensor(indices, dtype=torch.long)
        return torch.from_numpy(self.features[indices.numpy()])

    def get_keys(self, indices: Union[torch.Tensor, np.ndarray]) -> List[List[Optional[str]]]:
        """获取给定索引对应的键（image_file）"""
        if self.rev_key_mapping is None:
            return None

        indices = np.asarray(indices) if isinstance(indices, torch.Tensor) else indices
        return [[self.rev_key_mapping.get(idx, None) for idx in batch_indices] for batch_indices in indices]

    def retrieve_top_k(self, query_features: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        检索最相似的 k 个特征
        Args:
            query_features: 查询特征，形状为 [B, D] 或 [B, 1, D]
            
        Returns:
            Tuple[np.ndarray, np.ndarray, torch.Tensor]:
                - D: 相似度或距离，形状为 [B, K]
                - I: 索引位置，形状为 [B, K]
                - features: 检索到的特征，形状为 [B, K, D]
        """
        if query_features.dim() == 3:
            query_features = query_features.squeeze(1)  # [B, D]
        D, I = self.search(query_features, self.top_k_final)  # [B, K]
        retrieved_features = self.get_vectors(I)  # [B, K, D]
        # 将索引转换为图像路径
        top_k_image_paths = self.get_keys(I)
        return D, I, retrieved_features, top_k_image_paths
    
    def get_feature_by_image_file(self, image_file: str) -> Optional[torch.Tensor]:
        """通过 image_file 获取特征向量"""
        if self.key_mapping is None or self.features is None:
            logger.warning("Key mapping or features not initialized")
            return None

        feature_idx = self.key_mapping.get(image_file)
        if feature_idx is not None:
            return torch.from_numpy(self.features[feature_idx]).unsqueeze(0)
        else:
            logger.warning(f"Image file {image_file} not found in key_mapping")
            return None

    def retrieve_worst_k(self, query_features: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        检索最不相似的 k 个特征（即相似度最低）
        """
        if query_features.dim() == 3:
            query_features = query_features.squeeze(1)           

        k = self.top_k_final 
        D, I = self.search(query_features, self.top_k_final)  # D, I shape = [B, top_k_final]
        B = D.shape[0]

        final_D = np.zeros((B, k), dtype=np.float32)
        final_I = np.zeros((B, k), dtype=np.int64)

        for i in range(B):
            idxs = np.argsort(D[i])[:k]  # 最小相似度的前 k
            final_D[i] = D[i][idxs]
            final_I[i] = I[i][idxs]

        # 返回 shape [B, k, D] 的检索特征
        retrieved_features = self.get_vectors(final_I)  # shape: [B * k, D]
        retrieved_features = retrieved_features.reshape(B, k, -1)  # 修复点！
        
        return final_D, final_I, retrieved_features

