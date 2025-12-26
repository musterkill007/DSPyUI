import os
from dotenv import load_dotenv
import litellm

load_dotenv()

# 测试 gemini-3-pro-preview 模型
try:
    print("Testing gemini-3-pro-preview...")
    response = litellm.completion(
        model="gemini-3-pro-preview",
        messages=[{"role": "user", "content": "Hello, this is a test."}],
        max_tokens=50
    )
    print("✓ Success!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {str(e)}")
    print("\nTrying alternative model name: gemini/gemini-3-pro-preview")
    try:
        response = litellm.completion(
            model="gemini/gemini-3-pro-preview",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=50
        )
        print("✓ Success with gemini/ prefix!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e2:
        print(f"✗ Also failed: {type(e2).__name__}: {str(e2)}")
