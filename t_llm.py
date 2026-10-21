from config import settings
from openai import OpenAI

print("Base URL:", settings.llm_base_url)
print("Model:", settings.llm_model)
print("API Key configured:", bool(settings.llm_api_key))

client = OpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    timeout=30,
    max_retries=0
)

try:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": "Reply with OK."
            }
        ],
        max_tokens=50
    )

    print("Response:", response.choices[0].message.content)
    print("Usage:", response.usage)

except Exception as e:
    print("Error type:", type(e).__name__)
    print("Error:", str(e))

    if hasattr(e, "status_code"):
        print("HTTP Status:", e.status_code)

    if hasattr(e, "response"):
        print("Response Body:", e.response.text[:1000])