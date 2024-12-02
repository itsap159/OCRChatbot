import os
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
from typing import List, Dict, Tuple
import re
# External library imports
from dotenv import load_dotenv
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class PDFIngestion:
    def __init__(self, pdf_path: str, embedding_model: str = "text-embedding-3-large", index: str = 'qqapdf'):
        """
        Initialize the PDF ingestion pipeline
        
        Args:
            pdf_path (str): Path to the PDF file
            embedding_model (str): OpenAI embedding model to use
            index name (str) : Pinecone index name
        """
        self.pdf_path = pdf_path
        self.embedding_model = embedding_model
        self.index_name = index
        load_dotenv()
        self.vector_store = None
        # Initialize Pinecone client
        self.pinecone_client = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))
 

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pre-process image to improve OCR accuracy
        
        Args:
            image (np.ndarray): Input image to preprocess
        
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise and enhance
        denoised = cv2.fastNlMeansDenoising(threshold)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        return contrast_enhanced
    
    def perform_ocr(self) -> List[Dict[str, str]]:
        """
        Perform OCR on PDF
        
        Returns:
            List of dictionaries with page number and extracted text
        """
        ocr_text = []
        try:
            # Convert PDF to images
            images = convert_from_path(self.pdf_path)
            
            # Set Tesseract configuration
            os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tesseract"
            custom_config = r'''
            --oem 3 
            --psm 6 
            -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789áéíóúÁÉÍÓÚ&.,:;() 
            --dpi 300'''
            for i, image in enumerate(images):
                # Preprocess and OCR
                open_cv_image = np.array(image)
                if open_cv_image.shape[2] == 3:
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                preprocessed_image = self.preprocess_image(open_cv_image)
                page_text = pytesseract.image_to_string(preprocessed_image, config=custom_config, lang = 'spa')
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                ocr_text.append({
                    "page_number": i + 1, 
                    "text": page_text
                })
        
        except Exception as e:
            print(f"OCR Error: {e}")
        
        return ocr_text
    
    def create_pinecone_index(self):
        """
        Create Pinecone index if it doesn't exist
        """
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name, 
                dimension=3072,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
    
    def create_vector_store(self) -> Pinecone:
        """
        Create a Pinecone vector store from the PDF
        
        Returns:
            Pinecone vector store
        """
        # Ensure index exists
        self.create_pinecone_index()
        
        # Perform OCR
        pdf_text = self.perform_ocr()
        
        # Prepare text contents
        text_contents = [page['text'] for page in pdf_text]
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        # Split text using SemanticChunker
        text_splitter = SemanticChunker(embeddings,  breakpoint_threshold_type="gradient")
        
        # Split text with metadata
        texts = text_splitter.create_documents(
            text_contents, 
            metadatas=[{
                'source': self.pdf_path, 
                'page': page['page_number'],
                'chunk_id': f'chunk_{i}'
            } for i, page in enumerate(pdf_text)]
        )
        
        return Pinecone.from_texts(
            [t.page_content for t in texts], 
            embeddings, 
            metadatas=[t.metadata for t in texts],
            index_name=self.index_name
        )
