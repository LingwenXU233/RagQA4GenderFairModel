import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

def print_progress(step):
    print(f"[{time.strftime('%H:%M:%S')}] {step}")

# Step 1: Load the same Embedding Model
print_progress("Loading the embedding model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 2: Connect to Milvus Vector Database
print_progress("Connecting to Milvus and setting up vector store...")
index_params = {
    "index_type": "IVF_FLAT",  # Instead of "HNSW"
    "params": {"nlist": 512},  # Set nlist (higher = better recall)
    "metric_type": "L2"  # Supports "L2", "IP", "COSINE"
}

vectorstore = Milvus(
    embedding_function=embeddings,
    # collection_name="milvus_demo",
    connection_args={ "uri": "./milvus_demo.db",},  # Corrected connection args
    index_params=index_params
)
print_progress("Milvus vector store is ready.")

# Step 3: Load the Local LLM
print_progress("Loading Ollama LLaMA 3.1 model...")
llm = OllamaLLM(model="llama3.1:8b")
print_progress("LLM is loaded.")

# Step 4: Define Prompt Template

PROMPT_TEMPLATE = """
System: You are an AI ethics policy maker. You provide original answers from context infor enclosed in <context> tags when possible.
Also, you could provide explanation based on factual and statistical information when possible.
Use the following pieces of information to provide a accurate answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know.
<context>
{context}
</context>

<question>
{question}
</question>
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Step 5: Define RAG Pipeline
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print_progress("Setting up the RAG pipeline...")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print_progress("RAG pipeline is ready.")
#
# # Step 6: Run Query
query = "what can i do in the data preprocessing stage to ensure the gender equity?"
query2 = "what is z-score? "
print_progress(f"Running query: {query2}")
res = rag_chain.invoke(query2)
print_progress("Query completed.")

# Step 7: Display Result
print("\nQuery Result:")
print(res)
