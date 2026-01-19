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

"""
How do you create a tool using open ai agents sdk ? 
- We need to add a decorator on top of a function which we need to be exposed as a tool.
- The underlying implementation of the function will be called when the tool is invoked.
- The entire boilerplate JSON is automatically generated.

FunctionTool(name='send_email', 
             description='Send a test email using Mailjet', 
			 params_json_schema=
			    {'properties': {'body': {'title': 'Body', 'type': 'string'}}, 
				'required': ['body'], 
				'title': 'send_email_args', 
				'type': 'object', 
				'additionalProperties': False}, 
				on_invoke_tool=<function function_tool.<locals>._create_function_tool.<locals>._on_invoke_tool at 0x113f9e160>, 
				strict_json_schema=True, is_enabled=True)
"""

@function_tool
def send_email(body: str):
	"""
	Send a test email using Mailjet
	"""
	api_key=os.environ["MAILJET_API_KEY"]
	api_secret=os.environ["MAILJET_SECRET_KEY"]
	mailjet = Client(auth=(api_key, api_secret), version='v3.1')
	sender_email="nehalsateeshkumar@gmail.com"
	sender_name="Nehal"
	recipient_email="nehalsateeshkumar@gmail.com" 
	recipient_name="Nehal"
	content=body
	subject="Test email"
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


"""
Can we convert an agent into a tool ? 
- Yes we can
- By using the as_tool method provided by the Agent class.
"""


description = "Write a cold sales email"

# Convert agents to tools
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)
tools = [tool1, tool2, tool3, send_email]


instructions ="You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales_agent tools once before choosing the best one. \
You pick the single best email and use the send_email tool to send the best email (and only the best email) to the user."


sales_manager = Agent(name="Sales Manager", instructions=instructions, tools=tools, model="gpt-4o-mini")

message = "Send a cold sales email addressed to 'Dear CEO'"

with trace("Sales manager"):
    result = await Runner.run(sales_manager, message)

