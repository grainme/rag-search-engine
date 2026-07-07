# Precision asks: "How much of what you found is relevant?"
def precision_at_k(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    if len(retrieved_docs) == 0:
        return 0.0

    relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
    return relevant_retrieved / len(retrieved_docs)


# Recall asks: "How much of what's relevant did you find?"
def recall_at_k(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    if len(retrieved_docs) == 0:
        return 0.0

    relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
    return relevant_retrieved / len(relevant_docs)


# F1 encourages you to improve both metrics: a system that's great at precision but terrible at recall gets a lower F1 score.
def f1_score(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    recall = recall_at_k(retrieved_docs, relevant_docs)
    precision = precision_at_k(retrieved_docs, relevant_docs)

    return 2 * recall * precision / (precision + recall)
