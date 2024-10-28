from langchain_community.document_loaders import PyMuPDFLoader 
from langchain.text_splitter import CharacterTextSplitter 
from sentence_transformers import SentenceTransformer 
import numpy as np 
from pinecone import Pinecone, ServerlessSpec 


path = 'text.pdf' 
#loading document or reading document 

loader = PyMuPDFLoader(path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc.page_content))


#converting chucks into vector
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model if desired
vectors = model.encode(chunks, convert_to_tensor=True)

vectors_np = vectors.cpu().numpy()

pc = Pinecone(api_key='e729bbd8-652a-4c26-8804-a8934a2f67e9')  # Replace with your API key
index_name = "vector2"  # Choose an appropriate name for your index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=vectors_np.shape[1],  # Set the dimension to match vector size
        metric='euclidean',  # Choose an appropriate metric
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Specify the cloud and region
    )
index = pc.Index(index_name)
pinecone_data = [(str(i), vectors_np[i].tolist()) for i in range(len(vectors_np))]  # Create tuples of (id, vector)

print(f"Stored {len(pinecone_data)} vectors in Pinecone.")


