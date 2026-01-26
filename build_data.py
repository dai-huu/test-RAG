import os
import re
import pdfplumber
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PDF_PATH = "STSV2022Phan1.pdf"
EXTRACTED_MD = "extracted_content.md"
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
USE_MARKDOWN = True  # True: đọc từ markdown, False: đọc trực tiếp PDF


def load_from_markdown(md_path):
    """Đọc từ file markdown đã chỉnh sửa"""
    documents = []
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    pages = re.split(r'\n## 📄 Trang (\d+)\n', content)
    
    for i in range(1, len(pages), 2):
        if i + 1 < len(pages):
            page_num = int(pages[i])
            page_content = pages[i + 1].strip()
            
            if page_content:
                has_table = "📊 Bảng" in page_content or re.search(r'\|.*\|', page_content)
                
                documents.append(Document(
                    page_content=page_content,
                    metadata={
                        "source": md_path,
                        "page": page_num,
                        "has_table": has_table,
                        "content_type": detect_content_type(page_content),
                        "char_count": len(page_content)
                    }
                ))
    
    return documents


def extract_pdf_with_tables(pdf_path):
    """Trích xuất PDF với xử lý bảng"""
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"📄 Tổng số trang: {len(pdf.pages)}")
        
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Đọc PDF")):
            tables = page.extract_tables()
            table_texts = []
            
            for idx, table in enumerate(tables):
                if table and len(table) > 0:
                    table_str = f"\n\n[BẢNG {idx + 1}]\n"
                    
                    if table[0]:
                        header = [str(cell or "").strip() for cell in table[0]]
                        table_str += "| " + " | ".join(header) + " |\n"
                        table_str += "| " + " | ".join(["---"] * len(header)) + " |\n"
                        
                        for row in table[1:]:
                            if row:
                                cells = [str(cell or "").strip() for cell in row]
                                table_str += "| " + " | ".join(cells) + " |\n"
                    
                    table_texts.append(table_str)
            
            text = page.extract_text() or ""
            
            if table_texts:
                page_text = text + "\n" + "\n".join(table_texts)
            else:
                page_text = text
            
            page_text = clean_text_preserve_structure(page_text)
            
            if page_text.strip():
                documents.append(Document(
                    page_content=page_text,
                    metadata={
                        "source": PDF_PATH,
                        "page": page_num + 1,
                        "has_table": len(tables) > 0,
                        "table_count": len(tables),
                        "content_type": detect_content_type(page_text),
                        "char_count": len(page_text)
                    }
                ))
    
    return documents


def detect_content_type(text):
    """Phát hiện loại nội dung"""
    if "[BẢNG" in text or "📊 Bảng" in text:
        return "table"
    elif re.search(r'(Điều|ĐIỀU)\s+\d+', text):
        return "regulation"
    elif re.search(r'\d+\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', text):
        return "numbered_list"
    elif re.search(r'[-•]\s+', text):
        return "bullet_list"
    else:
        return "text"


def clean_text_preserve_structure(text):
    """Làm sạch text giữ cấu trúc"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*(Điều|ĐIỀU|Khoản|Mục|\d+\.|\[BẢNG)', line):
            cleaned_lines.append(line.strip())
        else:
            cleaned_lines.append(re.sub(r'\s+', ' ', line.strip()))
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\%\n\|°§]', '', text, flags=re.UNICODE)
    
    return text.strip()


def validate_chunks(splits):
    """Kiểm tra chunks"""
    issues = []
    for i, split in enumerate(splits):
        content = split.page_content
        
        if len(content) < 50:
            issues.append(f"⚠️  Chunk {i}: Quá ngắn ({len(content)} ký tự)")
        
        weird_ratio = len(re.findall(r'[^\w\s\.\,\-]', content)) / (len(content) + 1)
        if weird_ratio > 0.3:
            issues.append(f"⚠️  Chunk {i}: Nhiều ký tự lạ ({weird_ratio:.1%})")
    
    if issues:
        print("\n📊 Phát hiện một số vấn đề:")
        for issue in issues[:5]:
            print(issue)
        if len(issues) > 5:
            print(f"   ... và {len(issues) - 5} vấn đề khác")


def build_vectorstore():
    print("📚 Bắt đầu xây dựng vector database...\n")
    
    # 1. Đọc dữ liệu
    if USE_MARKDOWN:
        if not os.path.exists(EXTRACTED_MD):
            print(f"❌ Không tìm thấy {EXTRACTED_MD}")
            print(f"💡 Chạy 'python extract_pdf.py' trước")
            return
        
        print(f"📖 Đọc từ markdown: {EXTRACTED_MD}")
        docs = load_from_markdown(EXTRACTED_MD)
    else:
        if not os.path.exists(PDF_PATH):
            print(f"❌ Không tìm thấy {PDF_PATH}")
            return
        
        print(f"📖 Đọc từ PDF: {PDF_PATH}")
        docs = extract_pdf_with_tables(PDF_PATH)
    
    print(f"\n✅ Đọc được {len(docs)} trang")
    
    table_pages = sum(1 for d in docs if d.metadata.get("has_table", False))
    print(f"   📊 Số trang có bảng: {table_pages}")
    print(f"   📝 Tổng ký tự: {sum(d.metadata['char_count'] for d in docs):,}")

    # 2. Embedding
    print("\n🧠 Đang khởi tạo embedding...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 3. Chunking
    print("✂️  Đang chia nhỏ văn bản...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Tăng từ 1500 lên 2000
        chunk_overlap=300,  # Tăng từ 200 lên 300
        separators=[
            "\n## 📄 Trang",  # Không tách trang
            "\n### 📊 Bảng",  # Giữ bảng nguyên
            "\n#### Điều ",   # Giữ điều khoản nguyên
            "\n\n",           # Paragraph break
            "\n",             # Line break
            ". ",             # Sentence
            " "               # Word
        ],
        length_function=len,
    )
    splits = splitter.split_documents(docs)
    print(f"✅ Tạo được {len(splits)} chunks")

    # 4. Merge chunks quá ngắn
    MIN_CHUNK_SIZE = 100
    merged_splits = []
    i = 0
    
    while i < len(splits):
        current = splits[i]
        
        # Nếu chunk hiện tại quá ngắn, merge với chunk sau
        if len(current.page_content) < MIN_CHUNK_SIZE and i + 1 < len(splits):
            next_chunk = splits[i + 1]
            
            # Chỉ merge nếu cùng trang hoặc trang liền kề
            if abs(current.metadata.get('page', 0) - next_chunk.metadata.get('page', 0)) <= 1:
                merged_content = current.page_content + "\n" + next_chunk.page_content
                merged_chunk = Document(
                    page_content=merged_content,
                    metadata={
                        **current.metadata,
                        'merged': True,
                        'char_count': len(merged_content)
                    }
                )
                merged_splits.append(merged_chunk)
                i += 2  # Skip cả 2 chunks
                continue
        
        merged_splits.append(current)
        i += 1
    
    print(f"🔗 Merge thành {len(merged_splits)} chunks (từ {len(splits)})")
    splits = merged_splits

    # 5. Metadata
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
    
    validate_chunks(splits)

    # 6. Lưu Chroma
    print("\n💾 Đang lưu vector database...")
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
    
    BATCH_SIZE = 50
    vectorstore = None
    
    for i in tqdm(range(0, len(splits), BATCH_SIZE), desc="Lưu chunks"):
        batch = splits[i:i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(batch, embeddings, persist_directory=PERSIST_DIR)
        else:
            vectorstore.add_documents(batch)
    
    print(f"\n✅ Hoàn tất! Lưu {len(splits)} chunks tại: {PERSIST_DIR}")
    print(f"📦 Kích thước TB: {sum(len(s.page_content) for s in splits) // len(splits)} ký tự/chunk")


def main():
    build_vectorstore()


if __name__ == "__main__":
    main()
