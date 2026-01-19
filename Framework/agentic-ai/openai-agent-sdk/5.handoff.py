from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from typing import Dict
import os
import asyncio
from mailjet_rest import Client
# Agentic Workflow : Parallelization

# Create Agent Workflow
# Add dedicated system propmts
# Create dedicated agents

instructions1 = "You are a sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write professional, serious cold emails."

instructions2 = "You are a humorous, engaging sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write witty, engaging cold emails that are likely to get a response."

instructions3 = "You are a busy sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."

sales_agent1 = Agent(
        name="Professional Sales Agent",
        instructions=instructions1,
        model="gpt-4o-mini"
)

sales_agent2 = Agent(
        name="Engaging Sales Agent",
        instructions=instructions2,
        model="gpt-4o-mini"
)

sales_agent3 = Agent(
        name="Busy Sales Agent",
        instructions=instructions3,
        model="gpt-4o-mini",
)



description = "Write a cold sales email"

# Convert agents to tools
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

"""
What is a handoff ? how is it different from converting a agent to a tool ? 
- It is when an Agent collaborate with anothe agent
- The difference between a handoff and converting an agent to a tool is that,  
-  with handoff control passes across agent interaction
-  with tools control passes back to the original agent
"""


subject_instructions = "You can write a subject for a cold sales email. \
You are given a message and you need to write a subject for an email that is likely to get a response."

html_instructions = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model="gpt-4o-mini")
subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")

html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model="gpt-4o-mini")
html_tool = html_converter.as_tool(tool_name="html_converter",tool_description="Convert a text email body to an HTML email body")

# Dict[str, str] : why [str,str] ?  Because we are mapping string keys to string values
@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
	""" Send out an email with the given subject and HTML body to all sales prospects """
	api_key=os.environ["MAILJET_API_KEY"]
	api_secret=os.environ["MAILJET_SECRET_KEY"]
	mailjet = Client(auth=(api_key, api_secret), version='v3.1')
	sender_email="nehalsateeshkumar@gmail.com"
	sender_name="Nehal"
	recipient_email="nehalsateeshkumar@gmail.com" 
	recipient_name="Nehal"
	content=html_body
	subject=subject
	data = {
		'Messages': [
						{
							"From": {
									"Email": sender_email,
									"Name": sender_name
									},
							"To": [
									{
											"Email": recipient_email,
											"Name": recipient_name
									}
							],
							"Subject": subject,
							"TextPart": content,
					}
			]
	}
	result = mailjet.send.create(data=data)
	return {"status": f"{result.status_code}"}


tools = [subject_tool, html_tool, send_html_email]

instructions ="You are an email formatter and sender. You receive the body of an email to be sent. \
You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. \
Finally, you use the send_html_email tool to send the email with the subject and HTML body."

emailer_agent = Agent(
    name="Email Manager",
    instructions=instructions,
    tools=tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it")


tools = [tool1, tool2, tool3]
handoffs = [emailer_agent]

sales_manager_instructions = "You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales agent tools at least once before choosing the best one. \
You can use the tools multiple times if you're not satisfied with the results from the first try. \
You select the single best email using your own judgement of which email will be most effective. \
After picking the email, you handoff to the Email Manager agent to format and send the email."

sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=handoffs,
    model="gpt-4o-mini")

message = "Send out a cold sales email addressed to Dear CEO from Alice"

with trace("Automated SDR"):
    result = await Runner.run(sales_manager, message)