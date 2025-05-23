from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os
import requests
import uuid
import traceback
from dotenv import load_dotenv
import time

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask server is running!"

# Load environment variables
load_dotenv()

# Initialize ChromaDB and embedding model
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="chatpdf")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini API config
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"

# Debug endpoint
@app.route("/debug", methods=["GET"])
def debug():
    try:
        items = collection.get()
        return jsonify({
            "count": len(items["ids"]),
            "sample_metadata": items["metadatas"][0] if items["metadatas"] else None,
            "file_ids": list(set(m["fileId"] for m in items["metadatas"])) if items["metadatas"] else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ingest PDF endpoint
@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        print("📥 Received ingest request")
        data = request.get_json()
        print(f"📦 Request data: {data}")
        
        pdf_url = data.get("pdf_url")
        file_id = data.get("file_key")
        file_name = data.get("file_name")

        if not pdf_url or not file_id or not file_name:
            return jsonify({"error": "Missing pdf_url, file_key, or file_name"}), 400

        print(f"⬇️ Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to download PDF. Status: {response.status_code}"}), 400

        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        reader = PdfReader("temp.pdf")
        print(f"📄 PDF has {len(reader.pages)} pages")
        
        full_text = ""
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
        
        if not full_text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400

        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
        print(f"✂️ Split into {len(chunks)} chunks")
        
        embeddings = embedding_model.encode(chunks).tolist()
        print(f"🧬 Generated {len(embeddings)} embeddings")

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = str(uuid.uuid4())
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[doc_id],
                metadatas=[{"fileId": file_id, "fileName": file_name}]
            )
            if idx == 0:
                print(f"📝 First chunk preview: {chunk[:100]}...")

        return jsonify({
            "message": "✅ Documents ingested successfully",
            "details": {
                "chunks": len(chunks),
                "file_id": file_id,
                "file_name": file_name
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"❌ Ingestion failed: {str(e)}"}), 500

# Query endpoint
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        print("🔍 Query request:", data)

        query_text = data.get("query")
        file_id = data.get("file_key") or data.get("fileId")

        if not query_text or not file_id:
            return jsonify({"error": "Missing query or file_id"}), 400

        print(f"🔎 Searching context for file_id={file_id}")
        embedding = embedding_model.encode([query_text])[0]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
            where={"fileId": file_id}
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return jsonify({"response": "⚠️ No relevant context found."})

        context = "\n".join(documents)
        print(f"📚 Context found: {len(documents)} chunks")

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Context:\n{context}\n\nQuery:\n{query_text}\n\nAnswer:"
                }]
            }]
        }

        gemini_response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            params={"key": GOOGLE_API_KEY},
            json=payload
        )

        gemini_data = gemini_response.json()
        if "candidates" in gemini_data:
            reply = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({"response": reply})
        else:
            return jsonify({"response": f"❌ Gemini error: {gemini_data.get('error', {}).get('message', 'Unknown error')}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"❌ Query failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)