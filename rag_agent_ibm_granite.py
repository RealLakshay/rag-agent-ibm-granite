import sys
assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12 to run this notebook."

%pip install git+https://github.com/ibm-granite-community/utils \
    transformers \
    langchain_community \
    langchain_huggingface[full] \
    langchain_milvus \
    replicate \
    wget

%pip install milvus-lite

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

from langchain_milvus import Milvus
import tempfile
db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")
vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)

from langchain_community.llms import Replicate
from ibm_granite_community.notebook_utils import get_env_var
model_path = "ibm-granite/granite-3.3-8b-instruct"
model = Replicate(
    model=model_path,
    replicate_api_token=get_env_var('REPLICATE_API_TOKEN'),
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

import os
import wget
filename = 'state_of_the_union.txt'
url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'
if not os.path.isfile(filename):
  wget.download(url, out=filename)

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embeddings_tokenizer,
    chunk_size=embeddings_tokenizer.max_len_single_sentence,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)
doc_id = 0
for text in texts:
    text.metadata["doc_id"] = (doc_id:=doc_id+1)
print(f"{len(texts)} text document chunks created")

ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")

query = "What did the president say about Ketanji Brown Jackson?"
docs = vector_db.similarity_search(query)
print(f"{len(docs)} documents returned")
for doc in docs:
    print(doc)
    print("=" * 80)  # Separator for clarity

from ibm_granite_community.langchain import TokenizerChatPromptTemplate, create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# Create a Granite prompt for question-answering with the retrieved context
prompt_template = TokenizerChatPromptTemplate.from_template("{input}", tokenizer=tokenizer)
# Assemble the retrieval-augmented generation chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
)
rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
)

output = rag_chain.invoke({"input": query})
print(output['answer'])