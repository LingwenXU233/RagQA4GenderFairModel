import time
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from unstructured.cleaners.core import group_broken_paragraphs, clean_extra_whitespace


def print_progress(step):
    print(f"[{time.strftime('%H:%M:%S')}] {step}")


print_progress("Loading the documents")

loader = TextLoader(file_path="data/docs_md/doc_guidelines/NetHope_Community_Guidelines.md")
documents = loader.load()

document_content = documents[0].page_content

print_progress("Chunking the documents")
# text_splitter = MarkdownTextSplitter(chunk_size=5000, chunk_overlap=5000*0.1)
headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers= False) #  strip_headers= False
docs = text_splitter.split_text(document_content)

# Ensure metadata fields are correctly formatted
for doc in docs:
    if "languages" in doc.metadata:
        doc.metadata["languages"] = str(doc.metadata["languages"])  # Convert to string

# Load the embedding model
print_progress("Loading the embedding model")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Choose a supported index type
print_progress("Loading the vector database")
index_params = {
    "index_type": "IVF_FLAT",  # Instead of "HNSW"
    "params": {"nlist": 512},  # Set nlist (higher = better recall)
    "metric_type": "L2"  # Supports "L2", "IP", "COSINE"
}

vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={
        "uri": "./milvus_demo.db",
    },
    index_params=index_params,
    drop_old=True,  # Drop the old Milvus collection if it exists
)

print_progress("Calculating the similarity")
query = "key steps for data preprocessing stage "
query2 = "what is z-score?"
responses = vectorstore.similarity_search(query=query, k=5)
for response in responses:
    print("______________________________")
    print(response.page_content)
