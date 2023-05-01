from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS


loader = TextLoader('/home/toukoum/langChainBasic/axs.txt')
documents = loader.load()

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


print(len(docs))

embeddings = HuggingFaceEmbeddings()


# Vector Strore 
db = FAISS.from_documents(docs, embeddings)


query = "What's the new rear derailleur ?"
docs = db.similarity_search(query)
print(docs[0].page_content)