from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    # 1. Khởi tạo LLM Ollama (Đảm bảo bạn đã chạy 'ollama pull llama3')
    print("--- Đang kết nối với Ollama (Model: llama3)... ---")
    llm = OllamaLLM(model="llama3")

    # 2. Dữ liệu mẫu để máy học
    print("--- Đang chuẩn bị dữ liệu mẫu... ---")
    data = """
    Đồ án này là về hệ thống RAG chạy trên Ollama.
    Thực hiện đồ án này có là một nhóm có 3 sinh viên năm 4 gồm: Lương Cẩm Đào, Huỳnh Tấn Dương và Hồ Hữu Đại.
    Giảng viên hướng dẫn là Tiến sĩ Trịnh Tấn Đạt.
    Thời gian thực hiện đồ án là 7 tuần.
    Hệ thống sử dụng LangChain để kết nối và ChromaDB để lưu trữ vector.
    Mục tiêu là tạo ra một Chatbot có thể trả lời dựa trên tài liệu cá nhân.
    """

    # 3. Chia nhỏ văn bản (Chunking)
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    texts = text_splitter.split_text(data)

    # 4. Tạo Embedding Model (Tải model tí hon từ HuggingFace về máy)
    print("--- Đang khởi tạo Embedding (Lần đầu sẽ tải model khoảng 80MB)... ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Lưu vào Vector Database (Chỉ lưu tạm thời trong bộ nhớ để test)
    print("--- Đang đưa dữ liệu vào Vector DB... ---")
    vectorstore = Chroma.from_texts(texts, embeddings)

    # 6. Tạo quy trình RAG (LCEL - cách mới)
    template = """
        Bạn là trợ lý AI chỉ trả lời dựa trên ngữ cảnh được cung cấp.

        QUAN TRỌNG: 
        - CHỈ sử dụng thông tin từ ngữ cảnh bên dưới để trả lời
        - KHÔNG sử dụng kiến thức bên ngoài
        - Nếu ngữ cảnh KHÔNG chứa thông tin cần thiết, hãy trả lời: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
        - Chỉ trả lời bằng tiếng Việt.

        Ngữ cảnh:
        {context}

        Câu hỏi: {question}

        Trả lời:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()
    
    # Tạo chain với LCEL
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Vòng lặp hỏi đáp liên tục
    print("\n" + "="*60)
    print("🤖 Chatbot RAG đã sẵn sàng!")
    print("💡 Gõ 'exit', 'quit', hoặc 'thoát' để kết thúc")
    print("="*60 + "\n")
    
    while True:
        query = input("🙋 Bạn: ").strip()
        
        # Kiểm tra lệnh thoát
        if query.lower() in ['exit', 'quit', 'thoát', 'thoat']:
            print("\n👋 Tạm biệt! Hẹn gặp lại.")
            break
        
        # Bỏ qua nếu câu hỏi trống
        if not query:
            continue
        
        try:
            response = qa_chain.invoke(query)
            print(f"🤖 AI: {response}\n")
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}\n")

if __name__ == "__main__":
    main()