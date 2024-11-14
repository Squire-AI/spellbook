"""
Parts of Agent:
- planning node
- tool node
- reasoning node
- output format node

features:
- multimodal
- token counting
- tool calling
- observability
"""
import asyncio
from openai import AsyncOpenAI
from models import AppEnviron
from tools.models import Tool
from react import OpenAIReactAgent


environ: AppEnviron = AppEnviron()
client = AsyncOpenAI(api_key=environ.openai_api_key)


async def add_to_calendar(**kwargs) -> str:
    name = kwargs.get("name", "No Name Provided")
    description = kwargs.get("description", "No Description Provided")
    start_datetime = kwargs.get("start_datetime", "No Start DateTime Provided")
    end_datetime = kwargs.get("end_datetime", "No End DateTime Provided")
    location = kwargs.get("location", "No Location Provided")
    response = f"""Added the following to calendar: \n\t Event: {name}, Description: {description}, Start: {
        start_datetime}, End: {end_datetime}, Location: {location}"""
    print(response)
    return response


async def draft_to_email(**kwargs) -> str:

    pass


async def search_tool(**kwargs) -> str:
    return """Liverpool 3-1 Chelsea ,source_type:website source:ESPN source_url:https://espn.com """


async def calculator(**kwargs) -> str:
    return "110"


async def on_update(**kwargs) -> str:
    # print("run history", [value for _,
    #                       value in kwargs.items()][-1])
    pass

if __name__ == "__main__":
    agent = OpenAIReactAgent(
        debug=True,
        client=client,
        model="gpt-4o-mini",
        temperature=0.7,
        max_iterations=30,
        system_prompt="You are a helpful assistant",
        messages=[
            {
                "role": "user",
                "content": "add the following to my calendar, date night on 03/12/24 from 7-9, study 04/12/24 3-4pm "
            }
            # {"role": "user",
            #   "content": [{"type": "text", "text": "add this to my calendar"},
            #               {
            #       "type": "image_url",
            #       "image_url": {
            #           "url": "https://i.ibb.co/LSdX0RF/photo-6316392868339629238-y-1.jpg",
            #       },
            #   },]}
        ],
        tools=[
            Tool(
                id="abc",
                name="calculator",
                description="Calculator to calculate mathematical expressions",
                parameters={"type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "MathematicalExpression": {
                                    "type": "string",
                                    "description": "appropriate mathematical expressions"
                                }},
                            "required": [
                                "MathematicalExpression"
                            ]

                            },
                function=calculator
            ),
            Tool(
                id="xyz",
                name="search_tool",
                description="Search tool for searching the web for queries",
                parameters={"type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "search query string"
                                }},
                            "required": [
                                "query"
                            ]
                            },
                function=search_tool
            ),
            Tool(
                id="xyz",
                name="add_event_to_calendar",
                description="adds a single calendar event to calendar",
                parameters={"type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "name of the event"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "description of the event"
                                },
                                "start_datetime": {
                                    "type": "string",
                                    "description": "start datetime in ISO format"
                                },
                                "end_datetime": {
                                    "type": "string",
                                    "description": "end datetime in ISO format"
                                },
                                "location": {
                                    "type": "string",
                                    "description": "location of event, if nothing, leave as empty string"
                                }},
                            "required": [
                                "query"
                            ]
                            },
                function=add_to_calendar
            )
        ],
        on_step_update=on_update
    )

    async def run():
        response = await agent.run()
        print(response)
    asyncio.run(run())
