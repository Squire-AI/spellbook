from typing import Any, Dict, Literal, Union
from pydantic import BaseModel


class ReactChoiceOutput(BaseModel):
    choice: Union[Literal["THOUGHT"], Literal["ACTION"],
                  Literal["ANSWER"], Literal["PAUSE"]]
    prompt: str
