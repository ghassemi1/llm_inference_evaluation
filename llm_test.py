import os

from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    # base_url="http://YOUR-PC-IP:8000/v1",   # ← Replace with the real IP of the PC
    # base_url = "http://localhost:8000/v1",
    # base_url="http://172.26.149.221:8000/v1",
    base_url=os.getenv(
        "OPENAI_BASE_URL", "http://8867-173-34-61-14.ngrok-free.app/v1"
    ),
    api_key=os.getenv("OPENAI_API_KEY", "dummy123"),
)

response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "qwen2.5-3b"),
    messages=[{"role": "user", "content": "what is llm?"}],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.content)


##########################################

# from openai import OpenAI
# import httpx  # ← Add this import

# client = OpenAI(
#     base_url="https://7be6-173-34-61-14.ngrok-free.app/v1",
#     api_key="dummy123",
#     http_client=httpx.Client(verify=False),  # ← This fixes the SSL error
# )

# response = client.chat.completions.create(
#     model="qwen2.5-3b",
#     messages=[{"role": "user", "content": "what is tensorrt-llm?"}],
#     temperature=0.7,
#     max_tokens=1024,
# )

# print(response.choices[0].message.content)
