from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# To store sessions in memory
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a fresh history for a specific user session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]