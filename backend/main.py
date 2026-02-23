import logging
import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
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


tavily_fail_count = 0
tavily_circuit_open_until = 0

load_dotenv()
print("API KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)
client = OpenAI()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

logger = logging.getLogger("research-agent")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentAnswer(BaseModel):
    summary: str
    bullet_points: list[str]
    follow_up_questions: list[str]
    
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()

    response = await call_next(request)

    duration_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(f"request_id={request_id} path={request.url.path} status={response.status_code} duration_ms={duration_ms}")
    return response

class TopicRequest(BaseModel):
    topic: str
    
@app.get("/")
def root():
    return {"message": "Research Assistant Agent backend is running"}

def search_web(query: str):
    
    global tavily_fail_count, tavily_circuit_open_until

    # If circuit is open, skip call
    if time.time() < tavily_circuit_open_until:
        logger.warning("Tavily circuit open, skipping request")
        return "", []
    
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": query,
        "search_depth": "advanced",
        "max_results": 5
    }

    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=8
            )

            response.raise_for_status()
            data = response.json()
            tavily_fail_count = 0
            break  # success

        except requests.exceptions.Timeout:
            logger.warning(f"Tavily timeout (attempt {attempt+1})")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Tavily request error (attempt {attempt+1}): {e}")

        except ValueError:
            logger.warning(f"Invalid JSON from Tavily (attempt {attempt+1})")

        if attempt == max_retries:
            tavily_fail_count += 1
            logger.error(f"Tavily failed after retries. fail_count={tavily_fail_count}")

        if tavily_fail_count >= 3:
            tavily_circuit_open_until = time.time() + 60
            logger.error("Tavily circuit opened for 60 seconds")

            return "", []

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

def classify_intent(topic: str):
    prompt = f"""
    Classify the user input.

    Input: "{topic}"

    Return JSON only:
    {{"intent": "research"}} 
    OR
    {{"intent": "conversation"}}

    - "research" means the user wants research on a topic.
    - "conversation" means greeting, small talk, or casual message.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(resp.choices[0].message.content)

def decide_next_action(topic: str, memory_context: str):
    prompt = f"""
    You are an autonomous research agent.

    Topic: {topic}

    Context:
    {memory_context}

    You must respond in JSON only:

    {{
    "thought": "your reasoning about what to do next",
    "confidence": 0-1,
    "action": "search" or "answer",
    "query": "...",              # if search
    "final_answer_json": {{
    "summary": "string",
    "bullet_points": ["string", "string"],
    "follow_up_questions": ["string", "string", "string"]
    }} # if answer
    }}

    Rules:
    - Think about information gaps.
    - If missing evidence → search.
    - If enough evidence → answer.
    - Be adaptive.
    - If Observation Quality: weak → rewrite the query, do not reuse it.
    - final_answer_json MUST contain summary, bullet_points, follow_up_questions.
    - Do not use any other keys.
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

def build_academic_query(topic: str) -> str:
    t = topic.strip()
    return f"\"{t}\" paper DOI PDF arXiv ACM IEEE Springer"

@app.post("/research")
def research_topic(request: TopicRequest):
    # classify intent FIRST
    intent = classify_intent(request.topic)

    if intent["intent"] == "conversation":
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": request.topic}],
            temperature=0.7
        )

        return {
            "summary": resp.choices[0].message.content,
            "bullet_points": [],
            "follow_up_questions": [],
            "sources": []
        }
    memory_context = retrieve_similar_memory(request.topic)

    max_steps = 6
    step = 0
    final_output = None
    search_results = ""
    sources = []
    search_query = ""
    did_search = False

    while step < max_steps:

        try:
            action = decide_next_action(request.topic, memory_context)
            if "action" not in action:
                logger.error("Missing action key in agent response")
                return {"error": "Agent returned invalid structure"}
            
            if action.get("confidence", 0) < 0.6 and step < 3:
                action["action"] = "search"
            
            if action["action"] == "search" and "query" not in action:
                action["query"] = build_academic_query(request.topic)
        except Exception as e:
            logger.error(f"Invalid action JSON: {e}")
            return {"error": "Agent produced invalid action"}

        logger.info(f"STEP={step} ACTION={action.get('action')}")

        if action["action"] == "search":
            if "query" not in action:
                logger.error("Search action missing query")
                return {"error": "Agent search missing query"}
            search_query = action["query"]
            search_results, sources = search_web(search_query)

            # FIRST: check empty
            if not search_results.strip():
                logger.warning("No search results returned")
                break

            # THEN append to memory
            memory_context += f"""
            Previous Thought:
            {action.get("thought", "")}

            Search Query:
            {search_query}

            Observation:
            {search_results}
            """

            # THEN label quality
            if len(search_results.strip()) < 200 or len(sources) == 0:
                memory_context += "\nObservation Quality: weak\n"
            else:
                memory_context += "\nObservation Quality: ok\n"

        elif action["action"] == "answer":
            if "final_answer_json" not in action:
                logger.error("Answer action missing final_answer_json")
                return {"error": "Agent answer missing content"}
            final_output = action["final_answer_json"]
            break

        step += 1

    
    # After loop finishes
    if final_output is None:
        return {"error": "Agent failed to produce answer"}

    # Only now do reflection
    reflection = reflect_on_answer(
        request.topic,
        final_output,
        memory_context
    )

    if not reflection.get("approved", True):
        final_output = reflection["improved_answer"]
    
    try:
        validated = AgentAnswer(**final_output)
        final_output = validated.dict()
    except Exception as e:
        logger.error(f"Invalid agent JSON: {e}")
        return {"error": "Agent produced invalid structured output"}

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





