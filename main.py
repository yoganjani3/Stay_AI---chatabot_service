from backend.conversation.chat import chat_with_travel_assistant
# from backend.memory.chroma_memory.add_data import add_pdf_to_chroma
from backend.memory.mem0_memory.try_mem0 import add_memory_in_mem0, extract_relevant_memories

if __name__ == "__main__":
    # add_pdf_to_chroma(pdf_path=r"C:\Users\Admin\Downloads\StayAI-main\StayAI-main\jaipur.pdf")
    chat_with_travel_assistant()

    
