from openai import OpenAI

SYS_PROMPT = """
Your goal is to classify a given SENTENCE in the following order:
1) Generate a classification for the SENTENCE into one or more Sustainable Development Goals as a list. 
Think if the sentence relates to one or more SDGs. There can be many SDGs applicable to the sentence, but list only the most relevant ones.
If SENTENCE doesn't seem into any SDGs, just return 0 instead of a goal
2) Classify as True if the SENTENCE mentions Artificial Intelligence and related technologies, else False - append to the list.
3) Classify the sentiment of the text as Positive or Negative and append to the list.

Provide the answer strictly in the following format as a single List:
[SDGA, SDGB, ..., True/False, Positive/Negative]

example1 - [1, 5, True, Negative]
example2 - [11, 12, False, Negative]
example3 - [0, False, Positive]
"""


def get_classifications(client, sentence, model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYS_PROMPT
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
        max_tokens = 50
    )
    print(response)
    return response.choices[0].message.content.strip()

def create_batch_object(sentence: str, sentence_id: str, csv_path: str, model="gpt-4.1-mini"):

    splits = csv_path.split("/")
    batch_obj = {
        "custom_id": f"task-{sentence_id}-{splits[-3]}-{splits[-2]}",
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
            "max_output_tokens": 50
        }
    }

    return batch_obj

if __name__ == "__main__":
    client = OpenAI()
    sent = "We want to improve people\u2019s quality of life by preventing and combating disease (health), promoting educational equality, employability and economic participation (skills), and \nconserving natural resources (resources)."
    classification = get_classifications(client, sent, model="gpt-4.1-mini")
    print(classification)

