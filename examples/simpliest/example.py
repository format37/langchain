import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
import asyncio

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

# Define a Pydantic model for the tool arguments
class ReverseAndRepeatArgs(BaseModel):
    word: str = Field(description="The word that needs to be reversed")
    repeats: int = Field(description="The number of times the word should be repeated")

async def ReverseAndRepeat(word: str, repeats: int) -> str:
    reversed_word = word[::-1]
    return f"{reversed_word} " * repeats

async def main():
    chat_history = []
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4-0613", temperature=0)

    # Create a list of tools
    tools = []
    ReverseAndRepeatTool = StructuredTool.from_function(
            coroutine=ReverseAndRepeat,
            name="reverse_and_repeat",
            description="Reverses the provided word and repeats it the requested number of times.",
            args_schema=ReverseAndRepeatArgs,
            verbose=True,
        )
    tools.append(ReverseAndRepeatTool)

    # Define the prompt template with a more explicit system message
    system_message = SystemMessagePromptTemplate.from_template(
        """You are a helpful AI assistant."""
    )
    human_message = HumanMessagePromptTemplate.from_template("{input}")
    ai_message = AIMessagePromptTemplate.from_template("{agent_scratchpad}")
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message,
        ai_message
    ])

    # Create the agent
    chat_agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=chat_agent, tools=tools, verbose=True)
    message_text = "Hi, I'm Alice. Can you reverse my name and repeat it 3 times?"
    response = await agent_executor.ainvoke(
        {
            "input": message_text,
            "chat_history": chat_history,
        }
    )

    print(response['output'])

if __name__ == "__main__":
    asyncio.run(main())
    