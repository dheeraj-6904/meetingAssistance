import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

response = client.chat.complete(
    model="voxtral-small-2507",
    messages=[
        {"role": "user", "content": "Give me a 3-word greeting."}
    ],
)

print(response)