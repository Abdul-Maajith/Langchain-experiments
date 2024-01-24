from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# Remembers the entire context of the conversation 

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

def main():
    load_dotenv()

    llm = ChatOpenAI()
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationEntityMemory(llm=llm),
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        verbose=True
    )

    print("Hello, I am ChatGPT CLI!")

    while True:
        user_input = input("> ")

        ai_response = conversation.predict(input=user_input)

        print("\nAssistant:\n", ai_response)


if __name__ == '__main__':
    main()