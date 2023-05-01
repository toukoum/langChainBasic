from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain


from langchain.chains import SimpleSequentialChain


load_dotenv()

llm =  HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "minTokens":100})


# Prompt template
template  = """Explain me the concept of {subject}"""


prompt = PromptTemplate(
                        template=template, 
                        input_variables=["subject"]
                        )


chain = LLMChain(llm=llm, prompt=prompt)


#print(chain.run("Artificial Intelligence"))


template2  = """Turn the concept description of {subject} and explain it to me like I'm five """



prompt2 = PromptTemplate(
                        template=template, 
                        input_variables=["subject"]
                        )

chain2 = LLMChain(llm=llm, prompt=prompt2)


overall_chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)


explanation = overall_chain.run("Artificial Intelligence")
print(explanation)