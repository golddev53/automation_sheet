import os
import requests
import textwrap
import getpass
from flask import Flask, flash, redirect, jsonify
from config import *
from googlesearch import search
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from flasgger import LazyJSONEncoder

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
SERVER_ERROR = os.getenv("SERVER_ERROR", default=None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def OpenAI_Prompt(prompt):
    try:
        gptResponse = requests.post(
            'https://api.openai.com/v1/chat/completions',
            json={
                'model': 'gpt-3.5-turbo-16k',
                'messages': [
                    {
                        'role': 'system',
                        'content': """You are a simple search query generator. Generate 1 simple Search Query to use on Google that optimally and completely represents the Prompt below.

                        Make sure the Search Query is clear, accurate and concise.

                        Format: The Search Query should be output and nothing else."""
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 8000,
                'temperature': 0.5
            },
            headers={
                'Authorization': f'Bearer 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',
                'Content-Type': 'application/json',
            },
        )
        gptResponse.raise_for_status()
        return gptResponse.json()['choices'][0]['message']['content'].strip()
    except Exception as error:
        print(f"Error occurred: {error}")
        return 'False'

def get_top_5_urls(query):
    params = {
        "q": query,
        "num": 5,
        "api_key": os.getenv("SERPAPI_API_KEY", default=None)
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    top_5_urls = [result["link"] for result in results["organic_results"][:5]]

    return top_5_urls

def extract_data(page):
    soup = BeautifulSoup(page, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    return text

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# def co_rerank():
#     llm = OpenAI(temperature=0)
#     compressor = CohereRerank()
#     compression_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor, base_retriever=retriever
#     )

@app.route('/', methods=['GET'])
def get():
    return jsonify("hello")


@app.route('/serp', methods=['POST'])
def serp_search():
    prompt = request.form['question']
    print(prompt)
    response = OpenAI_Prompt(prompt)
    print(response)
    urls = get_top_5_urls(response)

    extracted_texts = []
    # for url in urls:
    page = requests.get("https://browsee.io/blog/exploring-exercises-and-the-best-resources-to-learn-reactjs/").text
    extracted_data = extract_data(page)
    # out = textwrap.wrap(extracted_data, 2034)
    extracted_texts.append(Document(extracted_data))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(extracted_texts)
    print("Oenai key---", OPENAI_API_KEY, texts)
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)).as_retriever(
        search_kwargs={"k": 20}
    )
    docs = retriever.get_relevant_documents(response)
    llm = OpenAI(temperature=0)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0), retriever=compression_retriever
    )

    chain({"query": response})
    # pretty_print_docs(docs)
    # print(extracted_data)

    return jsonify("post sucess")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)