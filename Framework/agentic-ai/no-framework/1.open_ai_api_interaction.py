# Load environment variables from a .env file

# Import the necessary library
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file, overriding existing ones
load_dotenv(override=True) 

# To get the OpenAI API key navigate to `https://platform.openai.com/signup`
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")


## Agentic Frameowrk : Prompt Chaining - Output from 1 LLM is passed as input to another LLM
openai = OpenAI()

question1 = "Can you pick a business area worth exploring for Agentic AI opprtunity?This should be just one phrase with one business area"
business_idea = [{'role':'user','content':question1}]

response1 = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=business_idea
)

# response1.choices[0].message.content : 0 because the api sometimes can provide more than one response
question2 = f"what is the pain point in {response1.choices[0].message.content}?provide something challenging ripe for an agentic solution?should be points  just a statement"
business_pain_point = [{'role':'user','content':question2}]

response2 = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=business_pain_point
)

question3 = f"what is the solution to these pain points{response2.choices[0].message.content}?provide something challenging ripe for an agentic solution?should pain points followed by a solution"
business_solution = [{'role':'user','content':question3}]

response3 = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=business_solution
)

print(f"1.{question1}:\n{response1.choices[0].message.content}")
print(f"2.{question2}:\n{response2.choices[0].message.content}")
print(f"3.{question3}:\n{response3.choices[0].message.content}")