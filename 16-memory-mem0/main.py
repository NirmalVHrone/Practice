import os
import openai
from mem0 import MemoryClient
import dotenv
dotenv.load_dotenv("../.env")

"""
Get a Mem0 API key here: https://mem0.dev/api-keys-avb

Ensure to export the MEM0_API_KEY environment variable.

```bash
export MEM0_API_KEY=your_key_here
```
"""

user_id = "NIRMAL"

memory = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
client = openai.Client()

messages = []


while True:
    user_input = input("User: ")
    while user_input.lower() == "":
        user_input = input("User: ")
        if user_input:
            break

    messages.append({"role": "user", "content": user_input})

    related_memories = memory.search(user_input, filters={"user_id": user_id})
    print("related_memories >>> ", related_memories)
    related_memories_text = ""

    if related_memories and "results" in related_memories and related_memories["results"]:
        related_memories_text = "\n - ".join([f"{m['memory']}" for m in related_memories["results"]])

    print("\n\n\nrelated_memories_text >>> ", related_memories_text)

    system_message = [
        {
            "role": "system",
            "content": f"""answer the user's question honestly.
Here are some relevant information you may find useful that previous interactions with the user has taught us:
{related_memories_text}
        """,
        }
    ]

    response = client.chat.completions.create(
        messages=system_message + messages,
        model="gpt-5-mini",
        reasoning_effort="minimal",
    )

    answer = response.choices[0].message.content

    messages.append({"role": "assistant", "content": answer})
    print(f"\nAssistant: {answer} \n")

    print("messages >>> ", messages)
    print("messages[-2:] >>> ", messages[-2:])
    memory.add(messages[-2:], user_id=user_id)