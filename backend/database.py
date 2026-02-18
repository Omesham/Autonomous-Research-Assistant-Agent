from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text


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
    summary = Column(Text, nullable=False)
    
Base.metadata.create_all(bind=engine)
