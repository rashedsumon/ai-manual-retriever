# utils/prompt_templates.py

def build_rag_prompt(retrieved_passages, question):
    """
    retrieved_passages: list of dicts with keys 'source' and 'text'
    """
    ctx = "\n\n---\n\n".join([f"Source: {r['source']}\n\n{r['text']}" for r in retrieved_passages])
    prompt = (
        "You are an assistant that answers the user's question using only the provided retrieved passages. "
        "If you cannot find the answer in the passages, reply: 'I don't know from the provided documents.'\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    return prompt
