"""
RAG Evaluation Framework
========================
評估 RAG 系統的答案質素，包含多個維度：
- Answer Relevance (答案相關性)
- Context Precision (上下文精確度)
- Context Recall (上下文召回率)
- Faithfulness (忠實度)
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.evaluation import load_evaluator
from langchain_core.prompts import PromptTemplate


@dataclass
class EvalResult:
    """單條問題的評估結果"""
    question: str
    expected_answer: str
    generated_answer: str
    contexts: List[str]
    answer_relevance: float
    context_precision: float
    context_recall: float
    faithfulness: float
    overall_score: float


class RAGEvaluator:
    """
    RAG 系統評估器

    Usage:
        evaluator = RAGEvaluator(groq_api_key="your-key")
        result = evaluator.evaluate_single(
            question="什麼是RAG?",
            expected_answer="RAG是檢索增強生成...",
            retrieved_contexts=["RAG stands for..."],
            generated_answer="RAG是一種AI技術..."
        )
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("需要提供 GROQ_API_KEY")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=self.groq_api_key,
            temperature=0.1,
            timeout=30
        )

        # 初始化 evaluators
        self.correctness_evaluator = load_evaluator(
            "labeled_criteria",
            criteria="correctness",
            llm=self.llm
        )

    def _evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        評估答案與問題的相關性 (0-1)
        用 LLM 判斷答案是否直接回應了問題
        """
        prompt = PromptTemplate.from_template("""
        你是一個嚴格的評分員。請評估以下答案對問題的相關程度。

        問題: {question}
        答案: {answer}

        請只輸出一個 0 到 1 之間的分數，其中：
        - 1.0 = 答案完全且直接回應了問題
        - 0.5 = 答案部分相關，但缺少重要資訊
        - 0.0 = 答案完全不相關

        只輸出數字，不要任何解釋。
        分數:
        """)

        chain = prompt | self.llm
        try:
            response = chain.invoke({"question": question, "answer": answer})
            score_text = response.content.strip()
            # 提取數字
            import re
            match = re.search(r'[\d.]+', score_text)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        return 0.0

    def _evaluate_context_precision(
        self, 
        question: str, 
        contexts: List[str], 
        expected_answer: str
    ) -> float:
        """
        評估檢索到的上下文有多精確 (0-1)
        有多少比例的 context chunks 對回答問題有幫助
        """
        if not contexts:
            return 0.0

        prompt = PromptTemplate.from_template("""
        你是一個嚴格的評分員。請評估以下檢索到的上下文對回答問題的精確度。

        問題: {question}
        預期答案: {expected_answer}

        檢索到的上下文片段:
        {contexts}

        請評估這些上下文片段中有多少比例對回答問題是有用的。
        只輸出一個 0 到 1 之間的分數。
        只輸出數字，不要任何解釋。
        分數:
        """)

        context_text = "\n\n---\n\n".join([f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)])

        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "question": question, 
                "expected_answer": expected_answer,
                "contexts": context_text
            })
            score_text = response.content.strip()
            import re
            match = re.search(r'[\d.]+', score_text)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        return 0.0

    def _evaluate_context_recall(
        self, 
        question: str, 
        contexts: List[str], 
        expected_answer: str
    ) -> float:
        """
        評估檢索是否覆蓋了回答問題所需的所有資訊 (0-1)
        """
        if not contexts:
            return 0.0

        prompt = PromptTemplate.from_template("""
        你是一個嚴格的評分員。請評估檢索到的上下文是否包含了回答問題所需的全部資訊。

        問題: {question}
        預期答案: {expected_answer}

        檢索到的上下文片段:
        {contexts}

        請評估這些上下文是否包含了回答問題所需的全部資訊。
        只輸出一個 0 到 1 之間的分數。
        只輸出數字，不要任何解釋。
        分數:
        """)

        context_text = "\n\n---\n\n".join([f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)])

        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "question": question, 
                "expected_answer": expected_answer,
                "contexts": context_text
            })
            score_text = response.content.strip()
            import re
            match = re.search(r'[\d.]+', score_text)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        return 0.0

    def _evaluate_faithfulness(
        self, 
        answer: str, 
        contexts: List[str]
    ) -> float:
        """
        評估答案是否忠實於上下文 (0-1)
        答案中有多少資訊可以在上下文中找到支持
        """
        if not contexts or not answer:
            return 0.0

        prompt = PromptTemplate.from_template("""
        你是一個嚴格的評分員。請評估以下答案是否忠實於提供的上下文。

        答案: {answer}

        參考上下文:
        {contexts}

        請評估答案中的資訊有多少比例可以在上下文中找到支持。
        如果答案包含了上下文中沒有的資訊（幻覺），請降低分數。
        只輸出一個 0 到 1 之間的分數。
        只輸出數字，不要任何解釋。
        分數:
        """)

        context_text = "\n\n---\n\n".join([f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)])

        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "answer": answer,
                "contexts": context_text
            })
            score_text = response.content.strip()
            import re
            match = re.search(r'[\d.]+', score_text)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        return 0.0

    def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        retrieved_contexts: List[str],
        generated_answer: str
    ) -> EvalResult:
        """評估單條問題"""

        relevance = self._evaluate_answer_relevance(question, generated_answer)
        precision = self._evaluate_context_precision(question, retrieved_contexts, expected_answer)
        recall = self._evaluate_context_recall(question, retrieved_contexts, expected_answer)
        faithfulness = self._evaluate_faithfulness(generated_answer, retrieved_contexts)

        # 綜合分數 (加權平均)
        overall = (
            relevance * 0.30 +
            precision * 0.25 +
            recall * 0.25 +
            faithfulness * 0.20
        )

        return EvalResult(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            contexts=retrieved_contexts,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            faithfulness=faithfulness,
            overall_score=overall
        )

    def evaluate_batch(
        self,
        dataset: List[Dict[str, Any]],
        rag_pipeline_func=None
    ) -> Dict[str, Any]:
        """
        批量評估

        dataset format:
        [
            {
                "question": "...",
                "expected_answer": "...",
                "retrieved_contexts": ["...", "..."],
                "generated_answer": "..."
            }
        ]
        """
        results = []

        for item in dataset:
            result = self.evaluate_single(
                question=item["question"],
                expected_answer=item["expected_answer"],
                retrieved_contexts=item.get("retrieved_contexts", []),
                generated_answer=item["generated_answer"]
            )
            results.append(result)

        # 計算平均分
        avg_relevance = sum(r.answer_relevance for r in results) / len(results) if results else 0
        avg_precision = sum(r.context_precision for r in results) / len(results) if results else 0
        avg_recall = sum(r.context_recall for r in results) / len(results) if results else 0
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results) if results else 0
        avg_overall = sum(r.overall_score for r in results) / len(results) if results else 0

        return {
            "total_questions": len(results),
            "average_scores": {
                "answer_relevance": round(avg_relevance, 3),
                "context_precision": round(avg_precision, 3),
                "context_recall": round(avg_recall, 3),
                "faithfulness": round(avg_faithfulness, 3),
                "overall": round(avg_overall, 3)
            },
            "detailed_results": [
                {
                    "question": r.question,
                    "answer_relevance": r.answer_relevance,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "faithfulness": r.faithfulness,
                    "overall_score": r.overall_score
                }
                for r in results
            ]
        }


def format_eval_report(results: Dict[str, Any]) -> str:
    """格式化評估報告為 Markdown"""
    lines = [
        "# 📊 RAG Evaluation Report",
        "",
        f"**Total Questions Evaluated:** {results['total_questions']}",
        "",
        "## Average Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Answer Relevance | {results['average_scores']['answer_relevance']:.1%} |",
        f"| Context Precision | {results['average_scores']['context_precision']:.1%} |",
        f"| Context Recall | {results['average_scores']['context_recall']:.1%} |",
        f"| Faithfulness | {results['average_scores']['faithfulness']:.1%} |",
        f"| **Overall** | **{results['average_scores']['overall']:.1%}** |",
        "",
        "## Detailed Results",
        "",
    ]

    for i, r in enumerate(results['detailed_results'], 1):
        lines.extend([
            f"### Q{i}: {r['question'][:60]}...",
            "",
            f"- Answer Relevance: {r['answer_relevance']:.1%}",
            f"- Context Precision: {r['context_precision']:.1%}",
            f"- Context Recall: {r['context_recall']:.1%}",
            f"- Faithfulness: {r['faithfulness']:.1%}",
            f"- **Overall: {r['overall_score']:.1%}**",
            "",
        ])

    return "\n".join(lines)


if __name__ == "__main__":
    # 簡單測試
    evaluator = RAGEvaluator()

    test_dataset = [
        {
            "question": "什麼是RAG?",
            "expected_answer": "RAG是檢索增強生成，結合檢索和生成模型。",
            "retrieved_contexts": [
                "RAG (Retrieval-Augmented Generation) 是一種結合檢索系統和生成模型的技術。",
                "RAG 通過從外部知識庫檢索相關資訊來增強語言模型的生成能力。"
            ],
            "generated_answer": "RAG是檢索增強生成技術，它結合了檢索系統和生成模型，從外部知識庫檢索資訊來增強回答。"
        }
    ]

    results = evaluator.evaluate_batch(test_dataset)
    print(format_eval_report(results))
