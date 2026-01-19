from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from typing import Dict
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
from pydantic import BaseModel

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

"""
1. What is AsyncOpenAI?
- It is an asynchronous client used to talk to an OpenAI compatible API without blocking event loop
- It is an asynchronous HTTP client wrapper built on top of asyncio
- Designed to be used with await, async for, streaming and concurrent requests 
- Yields control back to the event loop while waiting for the response
- Since agents can stream tokens, call tools, pause and resume and run concurrent agents an async client is used
- Returns co-routine object (await client.responses.create(...)) or async streams (async for event in ...)
- gemini_client = AsyncOpenAI(...), just creates async client no request is sent
"""

"""
2. What is OpenAIChatCompletionsModel?
-  A model adapter that teaches the Agents SDK:
   “How do I talk to a chat-completions-style model using this client?”
- It maps the generic agent SDK calls to the specific API calls of the underlying client
- It handles the differences in request/response formats between the agent SDK and the underlying client
"""


google_api_key = os.getenv('GOOGLE_API_KEY')
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client)

deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)

groq_api_key = os.getenv('GROQ_API_KEY')
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)
llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)



instructions1 = "You are a sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write professional, serious cold emails."

instructions2 = "You are a humorous, engaging sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write witty, engaging cold emails that are likely to get a response."

instructions3 = "You are a busy sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."


sales_agent1 = Agent(name="DeepSeek Sales Agent", instructions=instructions1, model=deepseek_model)
sales_agent2 =  Agent(name="Gemini Sales Agent", instructions=instructions2, model=gemini_model)
sales_agent3  = Agent(name="Llama3.3 Sales Agent",instructions=instructions3,model=llama3_3_model)

description = "Write a cold sales email"

tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("ed@edwarddonner.com")  # Change to your verified sender
    to_email = To("ed.donner@gmail.com")  # Change to your recipient
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

subject_instructions = "You can write a subject for a cold sales email. \
You are given a message and you need to write a subject for an email that is likely to get a response."

html_instructions = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model="gpt-4o-mini")
subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")

html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model="gpt-4o-mini")
html_tool = html_converter.as_tool(tool_name="html_converter",tool_description="Convert a text email body to an HTML email body")

email_tools = [subject_tool, html_tool, send_html_email]

instructions ="You are an email formatter and sender. You receive the body of an email to be sent. \
You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. \
Finally, you use the send_html_email tool to send the email with the subject and HTML body."


emailer_agent = Agent(
    name="Email Manager",
    instructions=instructions,
    tools=email_tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it")

tools = [tool1, tool2, tool3]
handoffs = [emailer_agent]

# Define structured output
class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str

"""
1. What is an input guardrail?
- Pre-execution safety/ policy 
- Runs before the main agent sees the user input
- Checks for allowed inputs, sensitive data or actions to block, modify or annotate the input
- Can halt execution, doesn't pollute agent prompt
"""

"""
2. Why input guardrails are agents themselves ? Why cant we just use regular expressions or functions?
- We cant use regex or functions because:
    1.Names are ambiguous i.e. they can refer to multiple people or entities
    2.Context matters(“Jordan”, “May”, “Will”)
    3.Multilingual inputs exist
    4.LLM's are better at semantic classification
- The LLM agent here has
  1. Narrow instructions
  2.Produces structured output
  3.Is cheaper/smaller (gpt-4o-mini)  
"""

guardrail_agent = Agent( 
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)

"""
3. What @input_guardrail does?
- It runs async function before main agent processes the user input.
- Function is automatically awaited
- Automatically passed:
  1. ctx -> shared execution context
  2. agent -> the main agent(not guardrail agent)
  3. message -> raw user input
"""

"""
4. What is ctx ? 
- It is a GuradrailContext
- It is created by Runner and not by us
- It exits for one logical run
- It contains trace/span id's, parent-child execution relationships, 
  telemetry hooks():
   - Telemetry : observability data about execution
   - Includes Traces(who ran, in what order),spans(how long each step took?), Errors, Metadata about runs
   - Telemetry hooks : functions that get called at specific points during execution to collect and report telemetry data
   - Here hooks are triggered when a run starts, a model call starts, a nested run starts, a run finishes with errors
   - Above in cte.context, it provides current trace, current span, parent span id
   - This indicates attach guardrail run as a child span of current run
   cancellation propagation:
   - If a parent run is cancelled, all child runs are also cancelled
   - This ensures that no unnecessary work is done if the main task is aborted
   - In this context guardrail failed
   timeout propagation
   - Scheduled timeouts for each run
   - If a parent run times out, all child runs are also timed out
   - This ensures that no unnecessary work is done if the main task is aborted
   shared metadata: 
   - non-prompt, non-output data about execution
   - run id, request id, user/session id, debug mode, trace flags
"""

"""
5. What woould happen if we dont include cte.context ? 
- New root trace is created
- Guardrail run appears as an unrelated execution
- Telemetry is fragmented
- Cancellation won’t propagate
- Timeouts won’t propagate
- Debugging becomes painful
"""

"""
6. What is the purpose of tripwire_triggered ? 
- It is a decision switch
- tripwire_triggered = True
  1. Cancels main agent execution
  2. Skips tools
  3. Skips model calls
  4. Returns a guardrail failure
- tripwire_triggered = False
  1. Continues to the main agent
  2. Passes the original message unchanged
"""

"""
7. What is full execution timeline for a guardrail run?
- User inputs arrive
- Input guardrails run
- guardrail agent is executed
- Structured output is returned
- Tripwire is checked
- Main agent runs (if allowed)
"""

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info={"found_name": result.final_output},tripwire_triggered=is_name_in_message)



sales_manager_instructions = "You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales agent tools at least once before choosing the best one. \
You can use the tools multiple times if you're not satisfied with the results from the first try. \
You select the single best email using your own judgement of which email will be most effective. \
After picking the email, you handoff to the Email Manager agent to format and send the email."


careful_sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=[emailer_agent],
    model="gpt-4o-mini",
    input_guardrails=[guardrail_against_name]
    )

message = "Send out a cold sales email addressed to Dear CEO from Alice"

with trace("Protected Automated SDR"):
    result = await Runner.run(careful_sales_manager, message) 