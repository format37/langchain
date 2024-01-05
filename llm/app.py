from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
"""from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain"""
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
import os

def main():
    # if 'OPENAI_API_KEY' not in os.environ: exit()
    if 'OPENAI_API_KEY' not in os.environ:
        print('Please, set OPENAI_API_KEY environment variable:\nexport OPENAI_API_KEY="..."')
        exit()

    # llm = ChatOpenAI(openai_api_key="...")
    llm = ChatOpenAI()

    print('Test 1: Ask a question')
    result = llm.invoke("how can langsmith help with testing?")
    print(type(result))
    print(result)

    print('Test 2: Ask a question with a prompt')
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    chain = prompt | llm 
    result = chain.invoke({"input": "how can langsmith help with testing?"})
    print(type(result))
    print(result)

    print('Test 3: Ask a question with a prompt and output parser')
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    result = chain.invoke({"input": "how can langsmith help with testing?"})
    print(type(result))
    print(result)

    print('\nDone!')


if __name__ == '__main__':
    main()
