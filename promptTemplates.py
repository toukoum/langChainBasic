from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain import PromptTemplate


load_dotenv()

llm =  HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "minTokens":100})


# Prompt template
template  = """write a {song} about {subject} """

prompt = PromptTemplate(
                        template=template, 
                        input_variables=["song", "subject"]
                        )


print(llm(prompt.format(song="song", subject="life")))