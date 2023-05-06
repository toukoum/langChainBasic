from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()


# llm = OpenAI(temperature=0)
#
# text = "what is the height of the White Mount ?"
#
# print(llm(text))

from langchain.schema import (
    HumanMessage, 
    SystemMessage
)
# from langchain.chat_models import ChatOpenAI
#
#
#
#
#
#Utilisation de base, similaire a celle de l'api
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl")
print(llm("hello"))
#
#
# #Pour faire une interface similaire à un chatbot
# chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
#
# message =[
#     SystemMessage(content="You are an expert in traduction"),
#     HumanMessage(content="Traduct this sentence in french : I am the best in every sports !")
# ]
#
# reponse = chat(message)
#
# print(reponse.content, end='\n')



