from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
import json
from database import SessionLocal, ResearchHistory

load_dotenv()
print("API KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)
client = OpenAI()

app = FastAPI()
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



@app.post("/research")
def research_topic(request: TopicRequest):

    # First LLM step: decide search query
    search_prompt = f"""
    You are generating a web search query.

    Return ONLY the search query string.
    Do not explain.
    Do not add formatting.
    Topic: {request.topic}
    """


    search_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0.3
    )

    search_query = search_response.choices[0].message.content

    # Call tool (search)
    search_results, sources = search_web(search_query)

    print("SEARCH QUERY:", search_query)
    # print("SEARCH RESULTS:", search_results)

    # Final LLM step: generate research answer using search results
    final_prompt = f"""
    Topic: {request.topic}

    Web Search Results:
    {search_results}

    Return a JSON object with the following structure:

    {{
    "summary": "...",
    "bullet_points": ["...", "..."],
    "follow_up_questions": ["...", "...", "..."]
    }}

    Do not include markdown.
    Return valid JSON only.
    """

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.7
    )
    result_text = final_response.choices[0].message.content

    
    parsed = json.loads(result_text)
    db = SessionLocal()

    new_entry = ResearchHistory(
        topic=request.topic,
        summary=parsed["summary"]
    )

    db.add(new_entry)
    db.commit()
    db.close()


    return {
    "summary": parsed["summary"],
    "bullet_points": parsed["bullet_points"],
    "follow_up_questions": parsed["follow_up_questions"],
    "sources": sources
}

@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(ResearchHistory).all()
    db.close()

    return [
        {
            "id": r.id,
            "topic": r.topic,
            "summary": r.summary
        }
        for r in records
    ]





