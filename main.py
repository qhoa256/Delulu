from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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

def is_valid_image(filename: str) -> bool:
    """
    Check if the file has a valid image extension
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

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
async def chat(
    user_id: str = Form(...),
    session_id: str = Form(...),
    text: str = Form(...)
):
    """
    Endpoint /chat nhận user_id, session_id, text dạng multipart/form-data.
    Có thể gọi bằng:
        curl -X POST http://localhost:8000/chat \
          -H 'accept: application/json' \
          -H 'Content-Type: multipart/form-data' \
          -F 'user_id=...' \
          -F 'session_id=...' \
          -F 'text=...'
    """
    try:
        db = SessionLocal()
        
        # Lấy các tin nhắn cũ cho user + session này
        previous_messages = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.session_id == session_id
        ).order_by(Conversation.created_at.desc()).all()

        # Tạo conversation_history (list dict)
        conversation_history = []
        for msg in reversed(previous_messages):
            conversation_history.extend([
                {"role": "user", "content": msg.text},
                {"role": "assistant", "content": msg.response}
            ])
        
        # Thêm tin nhắn hiện tại
        conversation_history.append({"role": "user", "content": text})

        # Prompt hệ thống
        system_prompt = (
            "You are a helpful AI assistant. Please provide concise and direct responses. "
            "Use the conversation history to maintain context and provide relevant answers."
        )

        # Ghép conversation_history vào full_prompt
        full_prompt = f"""
        System: {system_prompt}

        Conversation History:
        {' '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[:-1]])}

        Current question: {text}
        """

        # Tùy chỉnh tham số sinh nếu cần
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 200,
        }

        # Gọi mô hình Gemini
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        response_text = response.text

        # Lưu kết quả vào DB
        conversation = Conversation(
            user_id=user_id,
            session_id=session_id,
            text=text,
            response=response_text,
            conversation_history=str(conversation_history)
        )
        db.add(conversation)
        db.commit()
        db.close()

        # Trả về JSON
        return {
            "user_id": user_id,
            "session_id": session_id,
            "response": response_text
        }

    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=str(e))

async def is_question_about_person(text: str) -> bool:
    """
    Check if the question is about a person using Gemini model.
    Returns True if the question is about a person, False otherwise.
    """
    try:
        # System prompt to just return True or False
        system_prompt = """You are a question classifier. Your only task is to determine if the given text 
        contains a question about a person or people. Respond with ONLY "True" if the question is asking 
        about a person or people, or "False" if not. Do not include any other text or explanation in your response."""
        
        # Use gemini-2.0-flash-lite model for quick classification
        classifier_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Configure generation parameters
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 10,
        }
        
        # Send text to the model
        response = classifier_model.generate_content(
            f"{system_prompt}\n\nQuestion: {text}",
            generation_config=generation_config
        )
        
        response_text = response.text.strip().lower()
        
        # Return True if the response is "true", False otherwise
        return response_text == "true"
        
    except Exception as e:
        # Log the error but default to False in case of errors
        print(f"Error in person question classification: {str(e)}")
        return False

@app.post("/analyze-image")
async def analyze_image(
    user_id: str = Form(...),
    session_id: str = Form(...),
    text: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    
    try:
        # Lấy context từ các tin nhắn trước đó
        db = SessionLocal()
        previous_messages = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.session_id == session_id
        ).order_by(Conversation.created_at.desc()).all()

        # Tạo conversation history
        conversation_history = []
        for msg in reversed(previous_messages):
            conversation_history.extend([
                {"role": "user", "content": msg.text},
                {"role": "assistant", "content": msg.response}
            ])
        
        # Thêm tin nhắn hiện tại
        conversation_history.append({"role": "user", "content": text})
        
        # Tạo prompt với context
        system_prompt = """You are a helpful AI assistant. Please provide concise and direct responses.
        Use the conversation history to maintain context and provide relevant answers."""
        
        full_prompt = f"""
        System: {system_prompt}
        
        Conversation History:
        {' '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[:-1]])}
        
        Current question: {text}
        """

        # Configure generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 200,
        }
        
        # Check if question is about a person
        is_about_person = await is_question_about_person(text)
        
        # If question is about a person, return specific response
        if is_about_person:
            response_text = "Nguyễn Mạnh Hùng"
        else:
            # Process normally if not about a person
            # Xử lý response dựa trên việc có image hay không
            if image and image.filename:
                try:
                    # Kiểm tra định dạng file
                    if not is_valid_image(image.filename):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image format. Supported formats: jpg, jpeg, png, gif, bmp, webp"
                        )

                    # Đọc nội dung file ảnh
                    image_content = await image.read()
                    
                    # Kiểm tra xem file có nội dung không
                    if not image_content:
                        raise HTTPException(status_code=400, detail="Empty image file")

                    # Sử dụng vision model nếu có ảnh
                    vision_model = genai.GenerativeModel('gemini-2.0-flash')
                    response = vision_model.generate_content([
                        full_prompt,
                        {
                            "mime_type": image.content_type,
                            "data": image_content
                        }
                    ])
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing image: {str(e)}"
                    )
            else:
                # Sử dụng model text nếu không có ảnh
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
            response_text = response.text

        # Lưu conversation mới với history
        conversation = Conversation(
            user_id=user_id,
            session_id=session_id,
            text=text,
            response=response_text,
            conversation_history=str(conversation_history)
        )
        db.add(conversation)
        db.commit()
        db.close()

        return {
            "user_id": user_id,
            "session_id": session_id,
            "response": response_text
        }
    except Exception as e:
        if 'db' in locals():
            db.close()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/detect-face")
# async def detect_face(
#     user_id: str = Form(...),
#     session_id: str = Form(...),
#     text: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     try:
#         # Kiểm tra định dạng file
#         if not is_valid_image(image.filename):
#             raise HTTPException(
#                 status_code=400,
#                 detail="Invalid image format. Supported formats: jpg, jpeg, png, gif, bmp, webp"
#             )

#         # Đọc nội dung file ảnh
#         image_content = await image.read()
        
#         # Kiểm tra xem file có nội dung không
#         if not image_content:
#             raise HTTPException(status_code=400, detail="Empty image file")

#         # System prompt để chỉ trả về True hoặc False
#         system_prompt = """You are a face detection system. Your only task is to determine if the image contains 
#         the face of Nguyễn Hữu Quang Hòa. Respond with ONLY "True" if you see his face, or "False" if not. 
#         Do not include any other text or explanation in your response."""
        
#         # Sử dụng model gemini-2.0-flash-lite
#         face_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
#         # Gửi ảnh đến model
#         response = face_model.generate_content([
#             system_prompt,
#             {
#                 "mime_type": image.content_type,
#                 "data": image_content
#             }
#         ])
        
#         response_text = response.text.strip()
        
#         # Kiểm tra kết quả và trả về response tương ứng
#         if response_text.lower() == "true":
#             return {
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "name": "Nguyễn Hữu Quang Hòa",
#                 "detected": True
#             }
#         else:
#             # Gọi đến API chat nếu kết quả là False
#             chat_request = ChatRequest(
#                 user_id=user_id,
#                 session_id=session_id,
#                 text=text
#             )
            
#             # Sử dụng chat function để xử lý tiếp
#             chat_response = await chat(chat_request)
            
#             return {
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "detected": False,
#                 "response": chat_response["response"]
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 