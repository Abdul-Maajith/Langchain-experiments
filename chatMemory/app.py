import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

def main():
    llm = ChatOpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferMemory()
    )

    print("Hello, I am ChatGPT Assistant CLI!")

    while True:
        user_input = input("> ")
        ai_response = conversation.predict(input=user_input)
        print("\nAssistant:\n", ai_response, "\n")

if __name__ == '__main__':
    main()