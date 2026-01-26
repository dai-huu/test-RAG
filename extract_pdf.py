import os
import pdfplumber
from tqdm import tqdm

PDF_PATH = "STSV2022Phan1.pdf"
OUTPUT_PATH = "extracted_content.md"


def extract_pdf(pdf_path):
    """Trích xuất PDF ra markdown"""
    output = []
    output.append("# Nội dung PDF\n\n")
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"📄 Tổng số trang: {len(pdf.pages)}")
        
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Đọc PDF")):
            output.append(f"## Trang {page_num + 1}\n\n")
            
            # Text
            text = page.extract_text()
            if text:
                output.append(text + "\n\n")
            
            # Bảng
            tables = page.extract_tables()
            if tables:
                for idx, table in enumerate(tables):
                    if table:
                        output.append(f"**Bảng {idx + 1}:**\n\n")
                        for row in table:
                            if row:
                                cells = [str(c or "").replace('\n', ' ').strip() for c in row]
                                output.append("| " + " | ".join(cells) + " |\n")
                        output.append("\n")
            
            output.append("---\n\n")
    
    return "".join(output)


def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ Không tìm thấy {PDF_PATH}")
        return
    
    print("📚 Bắt đầu trích xuất...\n")
    content = extract_pdf(PDF_PATH)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ Hoàn tất! File: {OUTPUT_PATH}")
    print(f"📏 {len(content):,} ký tự")


if __name__ == "__main__":
    main()
