from typing import Any, Dict, Literal, Union
from pydantic import BaseModel


class ReactChoiceOutput(BaseModel):
    choice: Union[Literal["THOUGHT"], Literal["ACTION"],
                  Literal["COMPLETE"], Literal["OBSERVE"]]
    prompt: str
