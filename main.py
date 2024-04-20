from dotenv import load_dotenv

load_dotenv()

import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain


def extract_texts_from_data(data):
    pdfreader = PdfReader(data)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)


def get_data_for_augmentation_from_raw_data(raw_data_for_augmentation):
    texts_for_augmentation = extract_texts_from_data(raw_data_for_augmentation)
    return FAISS.from_texts(texts_for_augmentation,
                            embedding=HuggingFaceEmbeddings(
                                model_name="BAAI/bge-small-en-v1.5",
                                model_kwargs={'device': 'cpu'},
                                encode_kwargs={'normalize_embeddings': True}
                            ))


def ask_question(data_for_augmentation, query):
    llm = CTransformers(
        model=os.path.join(os.path.dirname(__file__), "transformers", "llama-2-7b-chat.Q4_0.gguf"),
        model_type="llama",
        config={'max_new_tokens': 300, 'temperature': 0.01, 'context_length': 1000})
    chain = load_qa_chain(llm)

    return chain.invoke({"input_documents": data_for_augmentation.similarity_search(query),
                         "question": query})


def main():
    raw_data = os.path.join(os.path.dirname(__file__), "data", "example.pdf")
    data_for_augmentation = get_data_for_augmentation_from_raw_data(raw_data)

    query = "What will Mr. Pallino most probably eat on wednesday?"

    result = ask_question(data_for_augmentation, query=query)
    print(result)

    print(f"The answer to the question is:\n{result['output_text']}")


if __name__ == '__main__':
    main()
