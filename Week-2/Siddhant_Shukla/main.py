from pydantic import BaseModel
from google import genai
from google.genai import types,Client

import requests
import os
from dotenv import load_dotenv
from typing import Callable, Any

load_dotenv()



#----------------------API REQUESTS---------------------------------------------------------------------

def get_country_info(country:str):
    try:
        res=requests.get(f"https://restcountries.com/v3.1/name/{country}", timeout=5)
        res.raise_for_status()
        data=res.json()[0]
        return {
            "name": data["name"]["common"],
            "capital": data["capital"][0],
            "population": data["population"],
            "region": data["region"]
        }
    except requests.RequestException as e:
        return f"Search failed: {e}"
        
    
def get_weather(city: str) -> str:
    try:
        res = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        res.raise_for_status()
        return res.text.strip()
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"
    

#---------------------------------------------Defining baseModels---------------------------------------------------------------------------
class WeatherInput(BaseModel):
    city :str


class CountryInput(BaseModel):
    country : str


# --------------------------------------------defining a tool class-------------------------------------------------------------------------

class Tool:
    def __init__(self, name:str , description : str , input_model : type[BaseModel] , func : Callable[...,str]):
        self.name=name
        self.description=description
        self.input_model=input_model
        self.func=func
    def run(self, raw_input : dict[str, Any]):
        validate= self.input_model(**raw_input)
        return self.func(**validate.model_dump())
    def to_gemini_def(self):
        return types.FunctionDeclaration (
            name = self.name,
            description = self.description,
            parameters=  self.input_model.model_json_schema()
        )
    
TOOLS : list[Tool] =[Tool("get_weather", "Get current weather for a city.", WeatherInput, get_weather),Tool("get_country_info", "Get information of the country.", CountryInput, get_country_info)]
TOOLS_map : dict[str, Tool]= {t.name: t for t in TOOLS}

class TravelAgent:
    def __init__(self):
        self.client = genai.Client()
        self.tools_definition = [
            types.Tool(
                function_declarations=[
                    t.to_gemini_def() for t in TOOLS
                ]
            )
        ]

    def run(self, user_message):
        messages=[types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )]
        while True:
            

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a travel suggestion agent. "
                        "Whenever country or weather information is needed "
                        "you must use the available tools."
                    ),
                    tools=self.tools_definition
                ),
                contents=messages
            )

            candidate = response.candidates[0]
            content = candidate.content

            # Save model response
            messages.append(content)

            function_call = None

            for part in content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_call = part.function_call
                    break

            # No tool call -> final answer
            if function_call is None:

                final_text = ""

                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text += part.text

                return final_text

            # Execute tool
            tool = TOOLS_map[function_call.name]

            result = tool.run(function_call.args)

            print(f"\nTool Called: {function_call.name}")
            print(f"Arguments: {function_call.args}")
            print(f"Result: {result}\n")

            # Send tool result back to Gemini
            messages.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=function_call.name,
                            response={
                                "result": result
                            }
                        )
                    ]
                )
            )
            
agent=TravelAgent()
print(agent.run("What is the weather in Tokyo right now?"))
print(agent.run("Give me information about the country Japan"))
