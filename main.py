import uvicorn
from fastapi import FastAPI
from datetime import datetime
# import history
import rag

app = FastAPI()

@app.get("/chat")
async def chat(message: str, session_id: str):
    # response = history.chat_with_system(message, session_id)['answer']
    response = rag.chat_with_system(message)['result']
    return {"message": response, "session_id": session_id, "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
