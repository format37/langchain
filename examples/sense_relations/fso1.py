import os
from pydantic import BaseModel, Field
from typing import Literal, List, Union, Any
from structed_output_extended import structure_output_completion
import logging
import sys

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a stream handler to print log messages to console
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Issue(BaseModel):
    description: str
    severity: Literal["low", "medium", "high"]

class ApplianceProblem(BaseModel):
    appliance_type: Literal["Oven", "Refrigerator", "Dishwasher", "Washing Machine"]
    brand: str
    issue: List[Issue]
    severity: Literal["low", "medium", "high"]

def main():
    os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

    user_input = """
    My name is Alex. My Samsung refrigerator has stopped working. It's no longer cooling the food, 
    which is a serious problem as all the food might spoil.
    """

    messages = [
        {
            "role": "user", 
            "content": 
            [
                {
                    "type": "text",
                    "text": f"Classify the following problem and return the result:\n{user_input}"
                }
            ]
        }
    ]

    response_format_representation = """
    class Issue(BaseModel):
        description: str
        severity: Literal["low", "medium", "high"]

    class ApplianceProblem(BaseModel):
        appliance_type: Literal["Oven", "Refrigerator", "Dishwasher", "Washing Machine"]
        brand: str
        issue: List[Issue]
        severity: Literal["low", "medium", "high"]
    """

    logger.info("[0] Calling completions for model: o1-mini")

    # Example usage for o1-mini
    o1_result = structure_output_completion(
        model="o1-mini",
        messages=messages,
        response_format_representation=response_format_representation,
        response_format=ApplianceProblem
    )

    logger.info("[0] o1-mini result:")
    if o1_result.parsed:
        logger.info(f"appliance_type: {o1_result.parsed.appliance_type}")
        logger.info(f"brand: {o1_result.parsed.brand}")
        logger.info(f"issue: {o1_result.parsed.issue}")
        logger.info(f"severity: {o1_result.parsed.severity}")
    else:
        logger.info(o1_result.refusal)
    
    logger.info("[0] Done.")

if __name__ == "__main__":
    main()
