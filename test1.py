import anthropic

client = anthropic.Anthropic()

messages = []

while True:
    user_input = input("You: ")

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=messages
    )

    assistant_text = response.content[0].text

    print("Claude:", assistant_text)

    messages.append({
        "role": "assistant",
        "content": assistant_text
    })