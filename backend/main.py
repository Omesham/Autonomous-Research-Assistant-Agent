from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
import json
from database import SessionLocal, ResearchHistory
import math
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
print("API KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)
client = OpenAI()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TopicRequest(BaseModel):
    topic: str
    
@app.get("/")
def root():
    return {"message": "Research Assistant Agent backend is running"}

def search_web(query: str):
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": query,
        "search_depth": "basic",
        "max_results": 5
    }

    response = requests.post(url, json=payload)
    data = response.json()

    results_text = ""
    sources = []

    for item in data.get("results", []):
        results_text += f"Title: {item['title']}\nContent: {item['content']}\n\n"
        sources.append({
            "title": item["title"],
            "url": item["url"]
        })

    return results_text, sources



def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1 * norm2)


def retrieve_similar_memory(topic, top_k=3):
    db = SessionLocal()
    records = db.query(ResearchHistory).all()

    if not records:
        db.close()
        return ""

    # Embed new topic
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=topic
    ).data[0].embedding

    scored = []

    for r in records:
        if r.topic.lower() == topic.lower():
            continue
        stored_embedding = json.loads(r.embedding)
        score = cosine_similarity(query_embedding, stored_embedding)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_records = scored[:top_k]

    memory_text = ""
    for score, r in top_records:
        memory_text += f"""
    Previous Topic: {r.topic}
    Summary: {r.summary}
    """
    print("Similarity scores:")
    for score, r in scored:
        print(r.topic, score)

    db.close()
    return memory_text

def decide_next_action(topic: str, memory_context: str):
    prompt = f"""
    You are a research agent.

    Topic: {topic}

    Relevant Past Research:
    {memory_context}

    Choose ONE action and return JSON only.

    Format:

    For search:
    {{"action": "search", "query": "..."}}

    For final answer:
    {{
    "action": "answer",
    "final_answer_json": {{
        "summary": "...",
        "bullet_points": ["...", "..."],
        "follow_up_questions": ["...", "...", "..."]
    }}
    }}

    Rules:
    - If you need fresh information, choose search.
    - If you are confident, choose answer.
    Return valid JSON only.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return json.loads(resp.choices[0].message.content)

def reflect_on_answer(topic, answer_json, memory_context):
    prompt = f"""
    You are reviewing a research answer.

    Topic: {topic}

    Answer:
    {json.dumps(answer_json)}

    Context Used:
    {memory_context}

    Evaluate the answer.

    Return JSON only:

    If answer is good:
    {{"approved": true}}

    If answer needs improvement:
    {{
    "approved": false,
    "improved_answer": {{
        "summary": "...",
        "bullet_points": ["...", "..."],
        "follow_up_questions": ["...", "...", "..."]
    }}
    }}

    Be strict but fair.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return json.loads(resp.choices[0].message.content)

@app.post("/research")
def research_topic(request: TopicRequest):

    memory_context = retrieve_similar_memory(request.topic)

    max_steps = 3
    step = 0
    final_output = None
    search_results = ""
    sources = []
    search_query = ""

    while step < max_steps:
        action = decide_next_action(request.topic, memory_context)
        print("STEP:", step)
        print("AGENT ACTION:", action)

        if action["action"] == "search":
            search_query = action["query"]
            search_results, sources = search_web(search_query)
            memory_context += f"\nSearch Results:\n{search_results}\n"
        
            if not search_results.strip():
                print("No search results, forcing stop.")
                break

        elif action["action"] == "answer":
            final_output = action["final_answer_json"]
            break

        step += 1

    reflection = reflect_on_answer(
    request.topic,
    final_output,
    memory_context
    )

    print("REFLECTION:", reflection)

    if not reflection.get("approved", True):
        print("Improving answer based on reflection.")
        final_output = reflection["improved_answer"]
    if final_output is None:
        return {"error": "Agent failed to produce answer"}
    # Generate embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=final_output["summary"]
    )

    summary_embedding = embedding_response.data[0].embedding

    # Save to DB
    db = SessionLocal()

    new_entry = ResearchHistory(
        topic=request.topic,
        search_query=search_query,
        summary=final_output["summary"],
        bullet_points=json.dumps(final_output["bullet_points"]),
        follow_up_questions=json.dumps(final_output["follow_up_questions"]),
        embedding=json.dumps(summary_embedding),
        sources=json.dumps(sources)
    )

    db.add(new_entry)
    db.commit()
    db.close()

    return {
        "summary": final_output["summary"],
        "bullet_points": final_output["bullet_points"],
        "follow_up_questions": final_output["follow_up_questions"],
        "sources": sources
    }

    # Save to DB (keep your embedding + storage logic here)
@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(ResearchHistory).all()
    db.close()

    return [
    {
        "id": r.id,
        "topic": r.topic,
        "search_query": r.search_query,
        "summary": r.summary,
        "bullet_points": json.loads(r.bullet_points),
        "follow_up_questions": json.loads(r.follow_up_questions),
        "sources": json.loads(r.sources),
        "created_at": r.created_at
    }
    for r in records
]





