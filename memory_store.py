import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key):
        # UPDATED: Using the exact model name found by your scanner
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
    def __call__(self, input: Documents) -> Embeddings:
        return self.embedder.embed_documents(input)

class MemoryModule:
    def __init__(self, api_key):
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.ef = GeminiEmbeddingFunction(api_key)
        self.collection = self.client.get_or_create_collection(
            name="twin_memory", 
            embedding_function=self.ef
        )

    def add_memory(self, user_input, agent_response):
        text = f"User: {user_input} | Agent: {agent_response}"
        import hashlib
        doc_id = hashlib.md5(text.encode()).hexdigest()
        self.collection.add(documents=[text], ids=[doc_id])

    def retrieve_context(self, query):
        results = self.collection.query(query_texts=[query], n_results=2)
        if results['documents'] and len(results['documents']) > 0:
            return "\n".join(results['documents'][0])
        return ""
