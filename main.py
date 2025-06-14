from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma

# Inicjalizacja modelu
llm = OllamaLLM(model="llama2")

# Ładowanie pliku CSV
loader = CSVLoader(file_path="car_prices.csv", encoding="utf-8", csv_args={"delimiter": ",", "quotechar": '"'})
documents = loader.load()
# print(f"Loaded {len(documents)} rows from CSV.")


# Tworzenie bazy wektorowej
embeddings = OllamaEmbeddings(model="llama2")
chroma_db = Chroma.from_documents(documents, embeddings)
# chroma_db.persist()

# Konfiguracja prompta
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given this context: {context}, please directly answer the question: {question}."
)

# Tworzenie łańcucha zapytań
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=chroma_db.as_retriever(search_kwargs={"k": 99}),  # Retrieve all 99 rows
    chain_type_kwargs={"prompt": prompt_template},
)


# Pętla rozmowy
print("Saburo is ready! Write 'exit' to close.")
while True:
    user_input = input("Question: ")
    
    if user_input.lower() == "exit":
        print("Saburo: See you!")
        break

    result = qa_chain.invoke({"query": user_input})
    # print("Saburo:", result["result"])
    print("\033[1mSaburo:\033[0m \033[1m" + result["result"] + "\033[0m")

