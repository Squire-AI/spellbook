REACT_PROMPT: str = (
    "{SystemPrompt}\n\n"
    "You run in a loop of Thought, Action, PAUSE, Observation.\n"
    "At the end of the loop you output an Answer\n"
    "Use Thought to describe your thoughts about the question you have been asked.\n"
    "Use Action to run one of the actions available to you - then return PAUSE.\n"
    "Observation will be the result of running those actions.\n\n"
    "Your available actions are:\n\n"
    "{Actions}\n\n"
    "Start by thinking step-by-step how you will solve this problem"
)
