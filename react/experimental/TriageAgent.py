"""experimental agents using OpenAI swarm"""
from swarm import Swarm, Agent


def update_calendar_event(event_id: str, name: str, start: str, end: str):
    print(f"Updating calendar event with ID: {event_id}")
    return f"""
updated {event_id}
name: {name}
start: {start}
end: {end}
"""


def search_calendar_events():
    print("Searching for calendar events")
    """
    Search for a calendar event by its ID.

    :param event_id: The ID of the event to search for.
    :return: Event details if found, otherwise None.
    """
    # Placeholder for search logic
    return """
    1. Event ID: event_1
       Title: Team Meeting
       Date: 2023-11-01
       Time: 10:00 AM

    2. Event ID: event_2
       Title: Project Deadline
       Date: 2023-11-05
       Time: 11:59 PM

    3. Event ID: event_3
       Title: Lunch with Client
       Date: 2023-11-10
       Time: 12:30 PM

    4. Event ID: event_4
       Title: Weekly Sync
       Date: 2023-11-15
       Time: 09:00 AM

    5. Event ID: event_5
       Title: Conference Call
       Date: 2023-11-20
       Time: 03:00 PM
    """


def create_calendar_event(event_name: str, start: str, end: str):
    print(f"Creating calendar event with name: {event_name}")
    """
    Create a new calendar event.

    : param event_details: A dictionary containing event details.
    : return: The ID of the created event.
    """
    # Placeholder for create logic
    return f"""
event name: {event_name}
start: {start}
end: {end}
"""


def delete_calendar_event(event_id: str):
    print(f"Deleting calendar event with ID: {event_id}")
    """
    Delete a calendar event by its ID.

    : param event_id: The ID of the event to delete.
    : return: True if deletion was successful, otherwise False.
    """
    # Placeholder for delete logic
    return f"deleted {event_id}"


def draft_email(to: str, draft: str):
    print(f"Drafting email to: {to}")
    print(f"""
to: {to}
draft: {draft}
""")


def transfer_to_triage(prompt: str):
    print(prompt)
    return triage_agent


def transfer_to_calendar_agent(prompt: str):
    print(prompt)
    return calendar_agent


def transfer_to_email_agent(prompt: str):
    print(prompt)
    return email_agent


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a triaging agent, you plan and assign tasks to the respective agents\n"
        "You have the following agents:\n"
        "Calendar Agent: Create, Update, Delete, Search Calendar events"
        "Email Agent: Draft emails"
        "When you complete the task, you are done with your task transfer to the most appropriate agent:"
        "- Triage Agent"
        "- Calendar Agent"
        "- Email Agent"
    ),
    functions=[transfer_to_calendar_agent,
               transfer_to_email_agent, transfer_to_triage]
)

calendar_agent = Agent(
    name="Calendar Agent",
    instructions=(
        "You are a calendar agent, you manage calendar events\n"
        "You have the following abilities:\n"
        "- Create calendar event\n"
        "- Update calendar event\n"
        "- Search calendar events\n"
        "- Delete calendar event\n"
        "When you complete the task, you are done with your task transfer to the most appropriate agent:"
        "- Triage Agent"
        "- Calendar Agent"
        "- Email Agent"
    ),
    functions=[create_calendar_event, update_calendar_event,
               delete_calendar_event, search_calendar_events,
               transfer_to_triage, transfer_to_email_agent, transfer_to_calendar_agent]

)

email_agent = Agent(
    name="Email Agent",
    instructions=(
        "You are an email agent, you manage my email"
        "You have the following abilities:"
        "- Draft email"
        "When you complete the task, you are done with your task transfer to the most appropriate agent:"
        "- Triage Agent"
        "- Calendar Agent"
        "- Email Agent"
    ),
    functions=[draft_email, transfer_to_triage,
               transfer_to_calendar_agent, transfer_to_email_agent]
)

client = Swarm()
messages = [{"role": "user",
             "content": "add breakfast 9pm on 20 nov 2024 to calendar, then draft an email to user@example.com"}]
response = client.run(agent=triage_agent, messages=messages, debug=True)
print(f"Response: {response.messages[-1]}")
