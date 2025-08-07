from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Model name for Ollama
model_name = "gemma3:4b"

# Load the model
model = ChatOllama(model=model_name)
prompt = ChatPromptTemplate.from_template("Question: {question} Answer:")

chain = prompt | model

history = {}
while True:
    user_input = input("\n Enter your question (or type 'exit' to quit): \n")
    if user_input.lower() == 'exit':
        print("Exiting the chat. Goodbye!")
        break
    response = chain.invoke({"question": user_input})
    print("\n **Generated Response:**")
    print(response.content)
    history['Question'] = user_input
    history['Answer'] = response.content

