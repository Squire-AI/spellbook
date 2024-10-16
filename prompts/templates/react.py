REACT_PROMPT: str = (
    "{SystemPrompt}\n\n"
    "You solve the problem by doing the following things:\n"
    "**THOUGHT**: Use Thought to describe your thoughts about the question you have been asked\n"
    "**ACTION**: Use Action to run one of the actions available to you\n"
    "**OBSERVE**: Use Observe to explain observed behaviour after actions\n"
    "***COMPLETE**: Use Complete when the task is completed\n\n"
    "**ACTIONS**:\n"
    "{Actions}\n"
    "Start by thinking through the steps using THOUGHT to achieve the task, use the relevant actions to achieve the goal"
)
