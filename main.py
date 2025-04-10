from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Optional
import base64

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# SQLite database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./conversations.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Conversation model
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    text = Column(Text)
    response = Column(Text)
    conversation_history = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic model for request
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    text: str

# Thêm Pydantic model cho image request
class ImageRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: Optional[str] = "Describe this image"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Lấy context từ các tin nhắn trước đó
        db = SessionLocal()
        previous_messages = db.query(Conversation).filter(
            Conversation.user_id == request.user_id,
            Conversation.session_id == request.session_id
        ).order_by(Conversation.created_at.desc()).all()

        # Tạo conversation history
        conversation_history = []
        for msg in reversed(previous_messages):
            conversation_history.extend([
                {"role": "user", "content": msg.text},
                {"role": "assistant", "content": msg.response}
            ])
        
        # Thêm tin nhắn hiện tại
        conversation_history.append({"role": "user", "content": request.text})
        
        # Tạo prompt với context
        system_prompt = """You are a helpful AI assistant. Please provide concise and direct responses.
        Use the conversation history to maintain context and provide relevant answers."""
        
        full_prompt = f"""
        System: {system_prompt}
        
        Conversation History:
        {' '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[:-1]])}
        
        Current question: {request.text}
        """

        # Configure generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 200,
        }

        # Get response from Gemini
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        response_text = response.text

        # Lưu conversation mới với history
        conversation = Conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            text=request.text,
            response=response_text,
            conversation_history=str(conversation_history)
        )
        db.add(conversation)
        db.commit()
        db.close()

        return {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "response": response_text
        }
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(request: ImageRequest, file: UploadFile = File(...)):
    try:
        # Đọc nội dung file ảnh
        image_content = await file.read()
        
        # Tạo model vision
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        
        # Tạo prompt cho vision model
        prompt = f"""Please analyze this image with the following request: {request.prompt}
        Provide a concise and direct response."""

        # Gọi Gemini Vision API để phân tích ảnh
        response = vision_model.generate_content([prompt, image_content])
        response_text = response.text

        # Lưu vào database
        db = SessionLocal()
        conversation = Conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            text=request.prompt,
            response=response_text
        )
        db.add(conversation)
        db.commit()
        db.close()

        return {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 