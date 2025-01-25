import json
import sys

import ollama
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from loguru import logger
from pydantic import BaseModel

from pdf_processor import extract_text_from_pdf


class QASystemConfig(BaseModel):
    document_pdf_path: str
    ollama_llm_name: str
    ollama_llm_temperature: float
    embedding_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_document: int = 1
    prompt: str


class QASystem:
    def __init__(self, config_path: str):
        with open(config_path, "r") as fp:
            config_data = json.load(fp)
        self.config = QASystemConfig(**config_data)
        logger.info(f"Loaded config from {config_path=}.")

        self.document_txt_path = extract_text_from_pdf(self.config.document_pdf_path)
        logger.info(
            f"Extracted text from {self.config.document_pdf_path} to {self.document_txt_path}."
        )

        self.retriever = self._build_retriever(
            document_path=self.document_txt_path,
            embedding_model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            top_k_documents=self.config.top_k_document,
        )
        logger.info("Built retriever.")

        self._pull_ollama_model(model_name=self.config.ollama_llm_name)
        logger.info("Successfully pulled model for ollama.")
        self._build_rag_chain(
            prompt_template=self.config.prompt,
            ollama_llm_name=self.config.ollama_llm_name,
            ollama_llm_temperature=self.config.ollama_llm_temperature,
        )
        logger.info("Succesfully built RAG chain.")

    def _pull_ollama_model(self, model_name: str):
        try:
            response = ollama.pull(model_name)
        except Exception as exception:
            logger.critical(
                f"ERROR! Failed to pull {model_name=}. Received {exception=}."
            )
            sys.exit(1)
        if response.status == "success":
            logger.info(f"Successfully pulled {model_name=}.")
        else:
            logger.critical(
                f"ERROR! Failed to pull {model_name=}. Response status = {response.status}."
            )
            sys.exit(1)

    def _build_retriever(
        self,
        document_path: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k_documents: int,
    ):
        loader = TextLoader(document_path)
        loaded_doc = loader.load()
        logger.info(f"Loaded document from {document_path=}")
        embedding = SentenceTransformerEmbeddings(
            model_name=embedding_model,
        )
        logger.info(f"Loaded embedding model from {embedding_model=}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(loaded_doc)
        logger.info("Made chunks.")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_documents})

        return retriever

    def _build_rag_chain(
        self,
        prompt_template: str,
        ollama_llm_name: str,
        ollama_llm_temperature: float,
    ):
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = OllamaLLM(model=ollama_llm_name, temperature=ollama_llm_temperature)

        # Chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Successfully build rag chain.")

    def invoke(self, question: str):
        return self.rag_chain.invoke(question)


if __name__ == "__main__":
    assistant = QASystem("base_config.json")
    assistant.build_rag_chain(
        assistant.config.prompt,
        assistant.config.ollama_llm_name,
        assistant.config.ollama_llm_temperature,
    )
    answer = assistant.invoke(
        "How many different norms can np.linalg.norm calculate? Give just a single number"
    )
    breakpoint()
    print(answer)
