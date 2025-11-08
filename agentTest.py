from dotenv import load_dotenv
import os
from google.adk.agents.llm_agent import Agent

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

def get_current_time(city: str) -> dict:
    return {"city": city, "time": "10:30 AM"}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Tells the current time in a specified city.",
    instruction="Use the get_current_time tool to provide accurate times.",
    tools=[get_current_time]
)


