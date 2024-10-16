from typing import List
from ..models import Tool

REACT_PLANNING_TOOLS: List[Tool] = [
    Tool(
        name="StepChoice",
        description="You think step by step, this is a single step and you can choose between, THOUGHT, OBSERVE, ACTION, COMPLETE",
        parameters={
            "type": "object",
            "properties": {
                    "choice": {
                        "type": "string",
                        "enum": [
                            "THOUGHT",
                            "OBSERVE",
                            "ACTION",
                            "COMPLETE"
                        ],
                        "description": "The most appropriate step to take in the current step-by-step reasoning, if the task has been satisfied use COMPLETE"
                    },
                "prompt": {
                        "type": "string",
                        "description": "The most appropriate & descriptive prompt that describes what is needed in the current stage"
                    }
            },
            "additionalProperties": False,
            "required": [
                "Thought"
            ]
        },
    )

]
