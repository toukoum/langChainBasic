from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain


from langchain.chains import SimpleSequentialChain


load_dotenv()

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl")


# Prompt template
template = """Act as if you were a human named Raphael Giraud who loves football. 
            You will have some questions and you must answer as if you were Raphael like . 
            Answer as if you were having a conversation with friends, using sentences
            Here is the question : {question}"""


prompt = PromptTemplate(
                        template=template, 
                        input_variables=["question"]
                        )


chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


print(chain.run("What's your favourite sport ?"))


# template2 = """Turn the concept description of {subject} and explain it to me like I'm five """
#
#
#
# prompt2 = PromptTemplate(
#                         template=template,
#                         input_variables=["subject"]
#                         )
#
# chain2 = LLMChain(llm=llm, prompt=prompt2)
#
#
# overall_chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)
#
#
# explanation = overall_chain.run("Artificial Intelligence")
# print(explanation)