from sqlalchemy import Column, Integer, String, Float

from database import Base


class EmotionHistory(Base):

    __tablename__ = "emotion_history"

    id = Column(Integer, primary_key=True, index=True)

    text = Column(String)

    emotion = Column(String)

    confidence = Column(Float)


class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String)

    email = Column(String, unique=True)

    password = Column(String)