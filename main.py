import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main():
    # Sử dụng qwen2.5:7b
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0)

    # 1. Kiểm tra vector DB đã được build chưa
    if not os.path.exists(PERSIST_DIR):
        print(f"❌ Không tìm thấy thư mục vector DB '{PERSIST_DIR}'. Hãy chạy build_data.py trước.")
        return

    # 2. Khởi tạo embeddings và nạp Chroma từ disk
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # 3. PROMPT CHO QWEN2.5 (Sử dụng format chuẩn, không dùng special tokens)
    template = """Bạn là trợ lý ảo thông minh của Trường Đại học Sài Gòn.
    Hãy sử dụng thông tin từ tài liệu dưới đây để trả lời câu hỏi của sinh viên.

    TÀI LIỆU:
    {context}

    CÂU HỎI: {question}

    YÊU CẦU:
    - Trả lời bằng tiếng Việt một cách tự nhiên và chính xác.
    - Chỉ dựa vào thông tin trong TÀI LIỆU để trả lời.
    - Nếu tài liệu không có thông tin, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin này trong Sổ tay sinh viên."
    - Tuyệt đối không tự bịa đặt thông tin.

    TRẢ LỜI:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Tăng số lượng context lấy ra với MMR để đa dạng hóa và tìm rộng hơn
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n🤖 Chatbot SGU (Qwen2.5) đã sẵn sàng!")
    
    while True:
        query = input("\n🙋 Bạn: ").strip()
        if query.lower() in ['exit', 'quit', 'thoát']: break
        
        try:
            # Lấy thông tin trang để kiểm tra
            context_docs = retriever.invoke(query)
            pages = set([str(d.metadata.get('page') + 1) for d in context_docs]) # +1 vì page bắt đầu từ 0
            print(f"🔍 Đang tìm ở trang: {', '.join(pages)}...")

            response = qa_chain.invoke(query)
            print(f"🤖 AI: {response}")
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()