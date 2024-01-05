from langserve import RemoteRunnable


def main():
    print("Test 1: Calling langserve api")
    remote_chain = RemoteRunnable("http://localhost:8000/agent/")
    result = remote_chain.invoke(
        {
            "input": "how can langsmith help with testing?",
            "chat_history": []
        }
    )
    print(type(result))
    print(result)
    print('\nDone!')


if __name__ == "__main__":
    main()
