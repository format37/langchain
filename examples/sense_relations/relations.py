from pydantic import BaseModel
import os
import openai
from openai import OpenAI
from datetime import datetime as dt

class Relation(BaseModel):
    object_a: str
    object_b: str
    explanation: str

class overview(BaseModel):
    relations: list[Relation]

os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")
client = OpenAI()

# Read tale from text file
filename = "time_storm"
with open(f"{filename}.txt", "r") as file:
    tale = file.read()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a literary critic. Your job is to determine a list of the most significant relations in the provided tale."},
        {"role": "user", "content": tale},
    ],
    response_format=overview,
)

message = completion.choices[0].message
report = ""
id = 0
if message.parsed:
    for relation in message.parsed.relations:
        report += f"{id}. [{relation.object_a}] - [{relation.object_b}]:\n{relation.explanation}\n\n"
        id += 1
else:
    print(message.refusal)

report_filename = f"{filename}_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
with open(report_filename, "w") as file:
    file.write(report)
