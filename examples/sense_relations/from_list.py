from pydantic import BaseModel, Field
from typing import Literal
import os
from openai import OpenAI

class ApplianceProblem(BaseModel):
    appliance_type: Literal["холодильник", "плита"]
    brand: Literal["Bayer", "Samsung", "LG", "Bosch"]
    issue: str
    severity: Literal["низкая", "средняя", "высокая"]

os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")
client = OpenAI()

user_input = """
У меня перестал работать холодильник Samsung. Он больше не охлаждает продукты, 
и это серьезная проблема, так как вся еда может испортиться.
"""

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Вы - эксперт по классификации проблем бытовой техники. На основе описания пользователя определите тип техники, бренд, проблему и ее серьезность."},
        {"role": "user", "content": f"Классифицируйте следующую проблему и верните результат:\n{user_input}"},
    ],
    response_format=ApplianceProblem,
)

message = completion.choices[0].message
if message.parsed:
    print(f"appliance_type: {message.parsed.appliance_type}")
    print(f"brand: {message.parsed.brand}")
    print(f"issue: {message.parsed.issue}")
    print(f"severity: {message.parsed.severity}")
else:
    print(message.refusal)
