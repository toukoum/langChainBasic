from dotenv import load_dotenv
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS


load_dotenv()

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_length":512})


loader = TextLoader('/home/toukoum/portfolio/src/context.txt')
documents = loader.load()

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


print(len(docs))

embeddings = HuggingFaceEmbeddings()


# Vector Strore 
db = FAISS.from_documents(docs, embeddings)

query = "Quel est ton sport ?"
docs = db.similarity_search(query)
similarity = docs[0].page_content
question = '{question}'

# Prompt template
template1 = """Act as if you are a human named Raphael Giraud. 
            You will have some questions and you must answer as if you were Raphael like . 
            Answer as if you were having a conversation with friends, using sentences
            Here is some information about Raphael Giraud that you need to write in your response : {context}
            Make sentence about this information
            Answer in the language of the question
            Here is the question : {question}"""

template = template1.format(context=similarity, question=question)


print(template)

prompt = PromptTemplate(
                        template=template,
                        input_variables=["question"]
                        )

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

print(chain.run(query))

# print(docs[0].page_content)
# print(chain.run(input_documents=docs, question=query))

