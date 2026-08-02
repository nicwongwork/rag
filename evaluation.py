from langchain.evaluation import load_evaluator

def evaluate_rag_answer(question: str, answer: str, context: str):
    """簡單評估RAG答案質素"""
    evaluator = load_evaluator("criteria", criteria="correctness")
    result = evaluator.evaluate_strings(
        prediction=answer,
        reference=context,
        input=question,
    )
    return result