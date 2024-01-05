from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
# from langchain.schema import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
import os

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        print('Please, set OPENAI_API_KEY environment variable:\nexport OPENAI_API_KEY="..."')
        exit()
    if 'TAVILY_API_KEY' not in os.environ:
        print('Please, set TAVILY_API_KEY environment variable:\nhttps://python.langchain.com/docs/integrations/retrievers/tavily\nexport TAVILY_API_KEY="..."')
        exit()
    # llm = ChatOpenAI(openai_api_key="...")
    # llm = ChatOpenAI()

    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = DocArrayInMemorySearch.from_documents(documents, embeddings)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    
    retriever = vector.as_retriever()
    
    # LangSmith offers several features that can help with testing:...

    # First we need a prompt that we can pass into an LLM to generate this search query

    '''prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    
    print('Test 1: Ask a question')
    result = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(type(result))
    print(result)

    print('Test 2: Ask a question with a prompt')
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    result = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(type(result))
    print(result)'''

    # Agent
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
    search = TavilySearchResults()
    tools = [retriever_tool, search]
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print('Test 1: Ask a question about LangSmith')
    result = agent_executor.invoke({"input": "how can langsmith help with testing?"})
    print(type(result))
    print(result)

    print('Test 2: Ask it about the weather')
    result = agent_executor.invoke({"input": "what is the weather in SF?"})
    print(type(result))
    print(result)

    print('Test 3: Conversations')
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    agent_executor.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })

    print('\nDone!')


if __name__ == '__main__':
    main()