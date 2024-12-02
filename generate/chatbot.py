from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain


def get_response(query: str, chat_history: List[Tuple], vector_store: Pinecone):
    """
        Generate a response using Retrieval-Augmented Generation (RAG)
        
        Args:
            query (str): User's input query
            chat_history (list): Previous conversation history
            Vector Store : Pinecone object
        Returns:
            dict: LLM Response
    """
    # simple_responses = {
    #         "thank you": "You're welcome!",
    #         "goodbye": "Goodbye! Have a great day!",
    #         "thanks": "You're welcome!",
    #         "bye": "Goodbye!"
    #     }
        
        # Normalize query
    # query = query.lower()
        
    #     # Handle simple responses
    # if query in simple_responses:
    #         return {
    #             "query": query,
    #             "result": simple_responses[query]
    #         }
        
        # Create a custom prompt template for lease document analysis
    prompt = PromptTemplate(
            template="""You are an expert in real estate lease analysis.
            Given the following context from a lease document:
            Context from Document: {context}
            
            and the previous conversations:
            Conversation History: {chat_history}
            
            Provide a precise, legally-informed answer to the following question:
            Question: {question}
            
            Guidelines:
            - Recognize social courtesies like 'hello', 'thank you', etc and respond accordingly.
            - Maintain conversation flow
            - Handle edge cases gracefully
            - Be concise and clear unless stated otherwise.
            - Recheck and reanalyze your answers based on the above given context and conversation history to give an accurate and well structured output.
            Answer in English.""",
            input_variables=["chat_history", "context", "question"]
        )

        # Configure language model
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Configure retriever
    retriever = vector_store.as_retriever(
            search_kwargs={
                'k': 10,  # Number of context chunks to retrieve
            }
        )

        # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        # Generate response
    result = chain.invoke({
            "question": query,
            "chat_history": chat_history
        })

        # Extract unique page numbers
    unique_pages = sorted(set(
            doc.metadata.get('page') 
            for doc in result['source_documents'] 
            if doc.metadata.get('page') is not None
        ))
        
    for doc in result['source_documents']:
            print(doc.metadata.get('page'))
        
        # Enhance answer with source information
    answer = result['answer']

    return {
            "query": query,
            "result": answer,
            "source_pages": unique_pages
        }