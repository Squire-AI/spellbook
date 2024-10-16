REACT_PROMPT: str = (
    "{SystemPrompt}\n\n",
    "You solve the problem by doing the following things:\n",
    "**THOUGHT**: Use Thought to describe your thoughts about the question you have been asked\n",
    "**ACTION**: Use Action to run one of the actions available to you\n",
    "***COMPLETE**: Use Complete when the task is completed\n\n",
    "**ACTIONS**:\n",
    "{Actions}\n",
    "Think through step by step, go:"
)
