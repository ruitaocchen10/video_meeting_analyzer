from google.adk.agents.llm_agent import Agent

'''
Be a Python function (not a class method unless wrapped)

Have clear type hints for all parameters

Return a JSON-serializable object (like a dict, list, str, or int)

Include a docstring explaining what it does

'''


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description="Tells the current time in a specified city.",
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time],
)
