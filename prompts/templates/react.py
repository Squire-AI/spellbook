REACT_PROMPT: str = (
    "{SystemPrompt}\n\n"
    "You operate in a cycle of Thought, Action, PAUSE, and Observation.\n"
    "At the conclusion of this cycle, you provide an Answer.\n"
    "Utilize Thought to articulate your considerations regarding the question posed.\n"
    "Employ Action to execute one of the available actions, followed by PAUSE.\n"
    "Observation reflects the outcomes of these actions.\n\n"
    "The actions you can perform are:\n\n"
    "{Actions}\n\n"
    "Begin by methodically contemplating how to address this problem."
)

REACT_ACTION_PROMPT: str = (
    "Action: {action_name}\n"
    "Action Description: {action_description}\n"
    "Action Fields:\n"
    "{action_fields}\n"
)

REACT_ACTION_FIELD_PROMPT: str = (
    " {action_field_name} \n"
    "   description:\n"
    "       {action_field_description} \n"
    "   type:\n"
    "       {action_field_type} "
)
