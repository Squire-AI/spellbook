

import re
from typing import Dict


def generate_prompt(template: str, variables: Dict[str, str]) -> str:
    """
    generates prompt based on prompt template and variables
    """
    validate_prompt_variables(template=template, variables=variables)
    return template.format(**variables)


def validate_prompt_variables(template: str, variables: Dict[str, str]) -> None:
    """
    validates if the variable is in the prompt
    """
    variables_in_prompt = re.findall(r'\{(.*?)\}', template)
    variables_in_prompt = [var for var in variables_in_prompt if isinstance(
        var, str) and re.match(r'^[A-Za-z0-9]+$', var)]
    variable_names = set(variables.keys())
    intersect = variable_names.intersection(set(variables_in_prompt))
    if len(variable_names) != len(intersect):
        remain = ",".join(variable_names-intersect)
        raise Exception(f"Prompt variable(s): {remain} is not within prompt")


if __name__ == "__main__":
    validate_prompt_variables("""
hello {world}
""", {"world": "world"})
