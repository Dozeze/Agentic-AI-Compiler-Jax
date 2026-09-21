import anthropic
import json
from concurrent.futures import ThreadPoolExecutor

client = anthropic.Anthropic()

# --- Real functions ---
def get_weather(city: str) -> str:
    fake_data = {"Stockholm": "14°C, cloudy", "Tokyo": "22°C, sunny"}
    return fake_data.get(city, "No data for that city")

def get_time(city: str) -> str:
    fake_times = {"Stockholm": "14:32 CET", "Tokyo": "22:32 JST"}
    return fake_times.get(city, "No timezone data for that city")

# Map tool name -> actual Python function
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_time": get_time,
}

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current local time for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
]

def execute_tool_call(block):
    """Run one tool_use block and return a properly formatted tool_result."""
    fn = TOOL_FUNCTIONS.get(block.name)
    if fn is None:
        result = f"Unknown tool: {block.name}"
    else:
        try:
            result = fn(**block.input)
        except Exception as e:
            result = f"Error running {block.name}: {e}"

    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": str(result)
    }

def run_conversation_turn(messages):
    """Calls Claude, and if it wants tools, runs them (in parallel) and loops until it answers in text."""
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return response  # Claude is done, just gave a text answer

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        for b in tool_use_blocks:
            print(f"Claude wants to call: {b.name}({b.input})")

        # Run all requested tool calls concurrently
        with ThreadPoolExecutor() as executor:
            tool_results = list(executor.map(execute_tool_call, tool_use_blocks))

        # All results go back in ONE user message
        messages.append({"role": "user", "content": tool_results})
        # loop continues — Claude may respond with text, or ask for more tools

messages = []

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})

    response = run_conversation_turn(messages)

    for block in response.content:
        if block.type == "text":
            print("Claude:", block.text)