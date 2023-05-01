from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv


from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.python import PythonREPL


load_dotenv()


llm =  HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "minTokens":100})


# tools = load_tools(["wikipedia", "llm-math"], llm=llm)
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# print(agent.run("What year did Lionel Messi Joined Barcelona? What is his current age raised to the 0.43 power?"))


agent_executor = create_python_agent(
                            llm=llm,
                            tool=PythonAstREPLTool(),
                            verbose=True
                                     )


agent_executor.run("Resolve this equation : 2x = 4")