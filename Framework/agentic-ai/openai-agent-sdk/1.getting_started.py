""" Why use open-ai agents SDK ? 
- Lightweight and Flexible
- Less constrained
- Less opinionated i.e. we can work with LLM models other than ones provided by openAI
- Makes common activities easy i.e. helps with easy tool calling
"""

""" 
Terminology: 
- Agent : Represent calls to LLM
- Handoff : Transfer of context or information between agents
- Guardrail : Represent constraints or limitations on agent behavior
"""

"""
Steps: 
- Create an agent instance
- Use with trace() to track interactions with an agent
- Call runner.run() to execute the agent
"""

# Pre-Requisite
# 1. Obtain open ai api key and store it in the .env file
# 2. OPENAI_API_KEY=<your_api_key>, format to store the key in the environement file

# Checks
# 1. Ensure the agent is created successfully
# 2. Verify the agent's instructions are set correctly
# 3. Confirm the agent can respond to user prompts


#GOAL:Create a agent to tell jokes

#TODO1:Create a agent name it Jokester and add system prompt(instructions)

#TODO2:Execute the agent by adding user prompt 


from dotenv import load_dotenv
from agents import Agent,Runner,trace
import asyncio
load_dotenv(override=True) # To reload environment variables from .env file

"""
This is how we create an agent
name : Name of the agent
instructions : This is like system prompt
model = Name of the LLM model

Output: 
Agent(
name='Jokester', instructions='You are a joke teller', 
prompt=None, handoff_description=None, 
handoffs=[], model='gpt-4o-mini', 
model_settings=
  ModelSettings(temperature=None, 
                top_p=None, 
                frequency_penalty=None, 
                presence_penalty=None, 
                tool_choice=None, 
                parallel_tool_calls=None, 
                truncation=None, 
                max_tokens=None, 
                reasoning=None, 
                metadata=None, 
                store=None, 
                include_usage=None, 
                extra_query=None, 
                extra_body=None, 
                extra_headers=None, 
                extra_args=None), 
    tools=[], 
    mcp_servers=[], 
    mcp_config={}, 
    input_guardrails=[], 
    output_guardrails=[], 
    output_type=None, 
    hooks=None, 
    tool_use_behavior='run_llm_again', 
    reset_tool_choice=True)
"""
agent = Agent(name="Jokester",instructions="You are a joke teller",model="gpt-4o-mini")

"""
Trace: It is a way to track all interactions with an LLM 

How to examine Trace ? 
- Navigate to the trace logs
- Look for the specific trace ID
- Analyze the input and output at each step

What happens when we execute Runner.run() ? 
- A co-routine object is created, hence we need to await to start event loop
- The instructions added here could be thought of as user prompt
 
"""
with trace("Telling a joke"):
    result = await Runner.run(agent,"Tell a joke about Autonomous AI Agents")
    print(result.final_output)
