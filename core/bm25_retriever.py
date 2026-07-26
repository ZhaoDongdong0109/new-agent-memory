"""
BM25 词法检索器

特点：
- 纯 Python 实现，无外部依赖
- 支持中文字符级分词 + 关键词提取
- 增量索引更新
- 适合中等规模数据（万级记忆碎片）
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class BM25Retriever:
    """
    BM25 词法检索器

    BM25 公式：
    score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    其中：
    - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    - f(qi, D) = 词频
    - |D| = 文档长度
    - avgdl = 平均文档长度
    - k1, b = 调节参数
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数（通常 1.2-2.0）
            b: 文档长度归一化参数（通常 0.75）
        """
        self.k1 = k1
        self.b = b

        # 索引结构
        self.doc_freqs: Dict[str, int] = {}  # term -> 包含该词的文档数
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> 文档长度
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: freq}
        self.total_docs: int = 0
        self.avg_doc_length: float = 0.0

        # 文档内容缓存（用于增量更新）
        self._doc_terms: Dict[str, List[str]] = {}  # doc_id -> terms

    def index(self, doc_id: str, text: str) -> None:
        """
        索引一个文档

        Args:
            doc_id: 文档 ID
            text: 文档文本
        """
        # 分词
        terms = self._tokenize(text)

        # 如果已存在，先删除旧索引
        if doc_id in self._doc_terms:
            self._remove_doc(doc_id)

        # 更新索引
        self._doc_terms[doc_id] = terms
        self.doc_lengths[doc_id] = len(terms)

        # 统计词频
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            if term not in self.term_freqs:
                self.term_freqs[term] = {}
            self.term_freqs[term][doc_id] = count
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # 更新统计信息
        self.total_docs = len(self._doc_terms)
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0

    def remove(self, doc_id: str) -> None:
        """
        删除一个文档的索引

        Args:
            doc_id: 文档 ID
        """
        self._remove_doc(doc_id)

    def _remove_doc(self, doc_id: str) -> None:
        """内部方法：删除文档索引"""
        if doc_id not in self._doc_terms:
            return

        terms = self._doc_terms[doc_id]
        term_counts = Counter(terms)

        for term, count in term_counts.items():
            if term in self.term_freqs and doc_id in self.term_freqs[term]:
                del self.term_freqs[term][doc_id]
                self.doc_freqs[term] = max(0, self.doc_freqs.get(term, 0) - 1)
                if self.doc_freqs[term] == 0:
                    del self.doc_freqs[term]
                    del self.term_freqs[term]

        del self._doc_terms[doc_id]
        del self.doc_lengths[doc_id]

        # 更新统计信息
        self.total_docs = len(self._doc_terms)
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            [(doc_id, score), ...] 按分数降序
        """
        if self.total_docs == 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # 计算每个文档的 BM25 分数
        scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            # IDF
            n = self.doc_freqs[term]
            idf = math.log((self.total_docs - n + 0.5) / (n + 0.5) + 1)

            # 对包含该词的每个文档计算分数
            for doc_id, tf in self.term_freqs[term].items():
                doc_len = self.doc_lengths[doc_id]

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_id] += idf * numerator / denominator

        # 排序并返回 top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词

        简单实现：
        1. 提取中文字符（2-gram）
        2. 提取英文单词
        3. 提取数字
        """
        terms = []

        # 提取中文字符（2-gram）
        chinese_chars = re.findall(r'[一-鿿]', text)
        for i in range(len(chinese_chars)):
            terms.append(chinese_chars[i])  # 单字
            if i + 1 < len(chinese_chars):
                terms.append(chinese_chars[i] + chinese_chars[i + 1])  # 2-gram

        # 提取英文单词（转小写）
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        terms.extend(english_words)

        # 提取数字
        numbers = re.findall(r'\d+', text)
        terms.extend(numbers)

        return terms

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            "total_docs": self.total_docs,
            "total_terms": len(self.doc_freqs),
            "avg_doc_length": self.avg_doc_length,
        }
