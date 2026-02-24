from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy import DateTime
from datetime import datetime

DATABASE_URL = "sqlite:///./research_memory.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # required for SQLite with FastAPI
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()




class ResearchHistory(Base):
    __tablename__ = "research_history"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    search_query = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    bullet_points = Column(Text, nullable=False)  # store JSON string
    follow_up_questions = Column(Text, nullable=False)  # store JSON string
    sources = Column(Text, nullable=False)  # store JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Text, nullable=False)  # store JSON string
    
Base.metadata.create_all(bind=engine)
