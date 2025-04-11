from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Connect to the database
SQLALCHEMY_DATABASE_URL = "sqlite:///./conversations.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def view_conversations():
    db = SessionLocal()
    try:
        # Query all conversations
        result = db.execute(text("SELECT * FROM conversations WHERE user_id = 'user123'"))
        
        print("\n=== Conversations in Database ===")
        for row in result:
            print(f"\nID: {row[0]}")
            print(f"User ID: {row[1]}")
            print(f"Session ID: {row[2]}")
            print(f"Text: {row[3]}")
            print(f"Response: {row[4]}")
            print(f"Created at: {row[5]}")
            print("-" * 50)
    finally:
        db.close()

if __name__ == "__main__":
    view_conversations() 