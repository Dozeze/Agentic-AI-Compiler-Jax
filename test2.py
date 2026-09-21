import anthropic
import json

client = anthropic.Anthropic()

# 1. Define the actual Python function
def get_weather(city: str) -> str:
    # In real life this would call a weather API
    fake_data = {"Stockholm": "14°C, cloudy", "Tokyo": "22°C, sunny"}
    return fake_data.get(city, "No data for that city")

# 2. Describe it to Claude so it knows the tool exists
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Stockholm"}
            },
            "required": ["city"]
        }
    }
]

messages = []

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Claude's reply gets added to history regardless of what's in it
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "tool_use":
        # Find the tool_use block(s) in the response
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"Claude wants to call: {block.name}({block.input})")

                # 3. Actually run the function
                if block.name == "get_weather":
                    result = get_weather(block.input["city"])
                else:
                    result = "Unknown tool"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # 4. Send the tool result(s) back as a user message
        messages.append({"role": "user", "content": tool_results})

        # Call Claude again so it can use the result to respond
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

    # Print whatever text Claude produced
    for block in response.content:
        if block.type == "text":
            print("Claude:", block.text)