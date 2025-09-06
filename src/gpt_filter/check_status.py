from openai import OpenAI

client = OpenAI()

# List all batches (default limit=20, can increase to 100)
batches = client.batches.list(limit=100)

for b in batches.data:
    if b.status == "in_progress":
        print(f"ID: {b.id} | Status: {b.status} | Created: {b.created_at} | Endpoint: {b.endpoint}")
