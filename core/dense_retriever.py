"""
Dense 向量检索器

特点：
- 纯 Python 实现，无外部依赖
- 使用简单 TF-IDF 向量作为默认 embedding
- 支持余弦相似度计算
- 可选接入外部 embedding API
"""

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple


class DenseRetriever:
    """
    Dense 向量检索器

    使用 embedding 函数将文本转换为向量，然后通过余弦相似度检索。

    默认使用简单 TF-IDF 向量（无需外部依赖），
    也可以通过 embedding_fn 参数接入外部 embedding API。
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        dimension: int = 384,
    ):
        """
        Args:
            embedding_fn: embedding 函数，接受文本返回向量
            dimension: 向量维度（默认 384）
        """
        self.embedding_fn = embedding_fn or self._default_embedding
        self.dimension = dimension

        # 向量索引
        self.vectors: Dict[str, List[float]] = {}  # doc_id -> vector
        self._doc_texts: Dict[str, str] = {}  # doc_id -> text (用于增量更新)

        # TF-IDF 统计（用于默认 embedding）
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0
        self._doc_term_counts: Dict[str, Counter] = {}

    def index(self, doc_id: str, text: str) -> None:
        """
        索引一个文档

        Args:
            doc_id: 文档 ID
            text: 文档文本
        """
        # 存储文本
        self._doc_texts[doc_id] = text

        # 计算 embedding
        vector = self.embedding_fn(text)
        self.vectors[doc_id] = vector

        # 更新 TF-IDF 统计
        terms = self._tokenize(text)
        self._doc_term_counts[doc_id] = Counter(terms)
        self._doc_count = len(self._doc_texts)
        self._update_idf()

    def remove(self, doc_id: str) -> None:
        """
        删除一个文档的索引

        Args:
            doc_id: 文档 ID
        """
        if doc_id in self.vectors:
            del self.vectors[doc_id]
        if doc_id in self._doc_texts:
            del self._doc_texts[doc_id]
        if doc_id in self._doc_term_counts:
            del self._doc_term_counts[doc_id]
        self._doc_count = len(self._doc_texts)
        self._update_idf()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            [(doc_id, cosine_sim), ...] 按相似度降序
        """
        if not self.vectors:
            return []

        # 计算查询向量
        query_vector = self.embedding_fn(query)

        # 计算余弦相似度
        results = []
        for doc_id, doc_vector in self.vectors.items():
            sim = self._cosine_similarity(query_vector, doc_vector)
            results.append((doc_id, sim))

        # 排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _default_embedding(self, text: str) -> List[float]:
        """
        默认 embedding 实现：TF-IDF 向量

        简单实现：
        1. 分词
        2. 计算 TF
        3. 乘以 IDF
        4. 归一化到固定维度
        """
        terms = self._tokenize(text)
        if not terms:
            return [0.0] * self.dimension

        # 计算 TF
        term_counts = Counter(terms)
        total_terms = len(terms)

        # 构建 TF-IDF 向量
        vector = [0.0] * self.dimension
        for term, count in term_counts.items():
            tf = count / total_terms
            idf = self._idf.get(term, 1.0)

            # 将 term 映射到向量维度
            idx = self._term_to_index(term)
            vector[idx] += tf * idf

        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def _term_to_index(self, term: str) -> int:
        """将 term 映射到向量维度索引"""
        # 使用 hash 映射到固定维度
        return hash(term) % self.dimension

    def _update_idf(self) -> None:
        """更新 IDF 统计"""
        if self._doc_count == 0:
            self._idf = {}
            return

        # 统计每个 term 出现在多少文档中
        doc_freq: Dict[str, int] = defaultdict(int)
        for term_counts in self._doc_term_counts.values():
            for term in term_counts:
                doc_freq[term] += 1

        # 计算 IDF
        self._idf = {}
        for term, freq in doc_freq.items():
            self._idf[term] = math.log((self._doc_count + 1) / (freq + 1)) + 1

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        terms = []

        # 中文字符（2-gram）
        chinese_chars = re.findall(r'[一-鿿]', text)
        for i in range(len(chinese_chars)):
            terms.append(chinese_chars[i])
            if i + 1 < len(chinese_chars):
                terms.append(chinese_chars[i] + chinese_chars[i + 1])

        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        terms.extend(english_words)

        # 数字
        numbers = re.findall(r'\d+', text)
        terms.extend(numbers)

        return terms

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            "total_docs": len(self.vectors),
            "dimension": self.dimension,
            "total_terms": len(self._idf),
        }
