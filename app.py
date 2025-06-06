import os
from flask import Flask, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TENANT_STORES = {}

@app.route("/upload", methods=["POST"])
def upload():
    data = request.json
    texts = data.get("texts")
    tenant = data.get("tenant_id")
    if not texts or not tenant:
        return {"error": "Missing texts or tenant_id"}, 400

    print(f"\n--- Upload Request ---")
    print(f"Tenant: {tenant}")
    print(f"Texts to embed:\n{texts}")

    embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print("Embedding texts using text-embedding-ada-002...")
    store = Chroma.from_texts(texts, embed, collection_name=f"store_{tenant}")
    TENANT_STORES[tenant] = store

    print("✅ Vectors stored in memory.")
    print(f"Tenant Store Keys: {list(TENANT_STORES.keys())}")
    return {"status": "uploaded", "tenant_id": tenant}

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("query")
    tenant = data.get("tenant_id")
    if not question or not tenant:
        return {"error": "Missing query or tenant_id"}, 400

    print(f"\n--- Query Request ---")
    print(f"Tenant: {tenant}")
    print(f"Query: {question}")

    if tenant not in TENANT_STORES:
        return {"error": "No data uploaded for this tenant"}, 404

    retriever = TENANT_STORES[tenant].as_retriever()
    print("Retrieving top matching documents using semantic search...")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa({"query": question})

    print("✅ Generated Response from LLM:")
    print(result["result"])

    print("\n📄 Retrieved Documents:")
    for doc in result["source_documents"]:
        print("•", doc.page_content)

    return jsonify({
        "response": result["result"],
        "retrieved_docs": [doc.page_content for doc in result["source_documents"]]
    })

if __name__ == "__main__":
    print("🚀 Server starting... Listening on http://localhost:5000")
    app.run(debug=True)
