import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class Metrics(Base):
    __tablename__ = 'metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    client_id = Column(String)
    round_number = Column(Integer)
    loss = Column(Float)
    roc_auc = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    drift_score = Column(Float)

class RoundInfo(Base):
    __tablename__ = 'round_info'

    round_number = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    clients_participated = Column(Integer)

def get_engine():
    db_url = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost:5432/federated_db")
    # For local testing without postgres, fallback to sqlite if specified
    if db_url.startswith("sqlite"):
        engine = create_engine(db_url)
    else:
        engine = create_engine(db_url)
    return engine

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
