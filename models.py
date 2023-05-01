from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

from langchain.schema import (
    HumanMessage, 
    SystemMessage
)
from langchain.chat_models import ChatOpenAI





#Utilisation de base, similaire a celle de l'api
llm =  HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "minTokens":100})
# print(llm("explain large language models in one sentence"))


#Pour faire une interface similaire à un chatbot
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

message =[
    SystemMessage(content="You are an expert in traduction"),
    HumanMessage(content="Traduct this sentence in french : I am the best in every sports !")
]

reponse = chat(message)

print(reponse.content, end='\n')
