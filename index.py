# Import the necessary modules
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# Corrected path to your PDF file and output text files
pdf_path = "text.pdf"
output_path = "output_chunks.txt"  # Path for the output text file
vector_output_path = "vector_data.txt"  # Path for the vector data file

# Load the PDF document
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Initialize the text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc.page_content))

# Write the chunks to a text file
with open(output_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"Chunk {i + 1}:\n{chunk}\n\n")  # Write chunk number and content

print(f"Chunks have been written to {output_path}.")

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model if desired

# Convert chunks into vector data
vectors = model.encode(chunks, convert_to_tensor=True)

# Optionally convert the tensor to a NumPy array for further processing
vectors_np = vectors.cpu().numpy()

# Write the vector data to a text file
with open(vector_output_path, "w", encoding="utf-8") as f:
    for i, vector in enumerate(vectors_np):
        vector_str = ', '.join(map(str, vector))  # Convert vector to a comma-separated string
        f.write(f"Vector {i + 1}: [{vector_str}]\n")  # Write vector number and values

print(f"Vector data has been written to {vector_output_path}.")

# Initialize Pinecone
pc = Pinecone(api_key='e729bbd8-652a-4c26-8804-a8934a2f67e9')  # Replace with your API key

# Check and create the index if it doesn't exist
index_name = "vector"  # Choose an appropriate name for your index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=vectors_np.shape[1],  # Set the dimension to match vector size
        metric='euclidean',  # Choose an appropriate metric
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Specify the cloud and region
    )

# Connect to the index
index = pc.Index(index_name)

# Prepare data for Pinecone
pinecone_data = [(str(i), vectors_np[i].tolist()) for i in range(len(vectors_np))]  # Create tuples of (id, vector)

# Upsert (update/insert) the vector data into Pinecone
index.upsert(pinecone_data)

print(f"Stored {len(pinecone_data)} vectors in Pinecone.")
