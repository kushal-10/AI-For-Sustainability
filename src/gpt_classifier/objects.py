from openai import OpenAI

SYS_PROMPT = """
You are given a SENTENCE to classify. Perform the following steps:

1. Symbolic vs Substantive Classification
   - SYMBOLIC = Mentions sustainability or AI in vague, aspirational, or rhetorical terms (e.g., commitments, values, goals) without evidence of concrete action, resources, or measurable outcomes.  
   - SUBSTANTIVE = Mentions sustainability or AI with clear evidence of implementation, resource allocation, measurable results, projects, or operational changes.  

2. Sentiment Classification
   - POSITIVE = The sentence expresses a beneficial, optimistic, or favorable view of sustainability/AI.  
   - NEGATIVE = The sentence expresses a critical, risk-oriented, or unfavorable view of sustainability/AI.  

Output Format (strict):  
Return a single Python-style list with exactly two elements:  
[SYMBOLIC/SUBSTANTIVE, POSITIVE/NEGATIVE]

Examples:  
- [SYMBOLIC, POSITIVE]  
- [SUBSTANTIVE, NEGATIVE]  
- [SUBSTANTIVE, POSITIVE]  
- [SYMBOLIC, NEGATIVE]  
"""


def create_batch_object(sentence: str, sentence_id: str, csv_path: str, model="gpt-4.1-mini"):

    splits = csv_path.split("/")
    batch_obj = {
        "custom_id": f"task||{sentence_id}||{splits[-3]}||{splits[-2]}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SYS_PROMPT
                },
                {
                    "role": "user",
                    "content": sentence
                }
            ],
            "max_tokens": 50
        }
    }

    return batch_obj

