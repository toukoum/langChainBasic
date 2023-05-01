from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain import ConversationChain


load_dotenv()


# model use
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "minTokens":100})
conversation = ConversationChain(llm=llm, verbose=True)

while True: 
    print('USER:')
    input_user = input()
    print(conversation.predict(input=input_user))
