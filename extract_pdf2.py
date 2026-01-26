"""
Trích xuất nội dung từ PDFs sang các file text
Hỗ trợ: TXT, JSON, Markdown
"""

import pdfplumber
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import re

class PDFExtractor:
    def __init__(self, pdf_dir: str = "./data/pdfs", 
                 output_dir: str = "./data/extracted_content"):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        
        # Tạo thư mục output nếu chưa có
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Làm sạch text tiếng Việt"""
        if not text:
            return ""
        
        # Xóa multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Xóa multiple newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Fix OCR errors
        text = text.replace('đ­', 'đ')
        text = text.replace('ư­', 'ư')
        text = text.replace('ơ­', 'ơ')
        
        return text.strip()
    
    def extract_page_data(self, page, page_num: int) -> Dict:
        """
        Trích xuất data từ 1 page
        """
        # Extract text
        text = page.extract_text() or ""
        text = self.clean_text(text)
        
        # Extract tables
        tables = page.extract_tables()
        
        # Format tables
        formatted_tables = []
        if tables:
            for table_idx, table in enumerate(tables, 1):
                formatted_table = {
                    'table_id': table_idx,
                    'rows': []
                }
                
                for row in table:
                    if row:
                        cleaned_row = [str(cell or "").strip() for cell in row]
                        formatted_table['rows'].append(cleaned_row)
                
                formatted_tables.append(formatted_table)
        
        return {
            'page': page_num,
            'text': text,
            'tables': formatted_tables,
            'has_tables': len(formatted_tables) > 0,
            'char_count': len(text)
        }
    
    def extract_pdf(self, pdf_path: Path) -> Dict:
        """
        Trích xuất toàn bộ PDF
        """
        print(f"⚙️  Đang trích xuất: {pdf_path.name}")
        
        pdf_data = {
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'extracted_at': datetime.now().isoformat(),
            'total_pages': 0,
            'pages': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_data['total_pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = self.extract_page_data(page, page_num)
                    pdf_data['pages'].append(page_data)
                
                # Statistics
                total_chars = sum(p['char_count'] for p in pdf_data['pages'])
                pages_with_tables = sum(1 for p in pdf_data['pages'] if p['has_tables'])
                total_tables = sum(len(p['tables']) for p in pdf_data['pages'])
                
                pdf_data['statistics'] = {
                    'total_characters': total_chars,
                    'pages_with_tables': pages_with_tables,
                    'total_tables': total_tables
                }
                
                print(f"   ✅ {pdf_data['total_pages']} trang, {total_tables} bảng, {total_chars:,} ký tự")
                
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            pdf_data['error'] = str(e)
        
        return pdf_data
    
    def save_as_txt(self, pdf_data: Dict, output_path: Path):
        """
        Lưu thành file TXT thuần
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"FILE: {pdf_data['filename']}\n")
            f.write(f"Trích xuất lúc: {pdf_data['extracted_at']}\n")
            f.write(f"Tổng số trang: {pdf_data['total_pages']}\n")
            
            if 'statistics' in pdf_data:
                stats = pdf_data['statistics']
                f.write(f"Số ký tự: {stats['total_characters']:,}\n")
                f.write(f"Số bảng: {stats['total_tables']}\n")
            
            f.write("="*80 + "\n\n")
            
            # Content
            for page_data in pdf_data['pages']:
                f.write(f"\n{'='*80}\n")
                f.write(f"TRANG {page_data['page']}\n")
                f.write(f"{'='*80}\n\n")
                
                # Text
                f.write(page_data['text'])
                f.write("\n\n")
                
                # Tables
                if page_data['tables']:
                    for table in page_data['tables']:
                        f.write(f"\n--- BẢNG {table['table_id']} ---\n\n")
                        
                        for row_idx, row in enumerate(table['rows']):
                            f.write(" | ".join(row) + "\n")
                            
                            # Separator after header
                            if row_idx == 0:
                                f.write("-" * 80 + "\n")
                        
                        f.write("\n")
        
        print(f"   💾 Đã lưu: {output_path.name}")
    
    def save_as_json(self, pdf_data: Dict, output_path: Path):
        """
        Lưu thành JSON (structured data)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 Đã lưu: {output_path.name}")
    
    def save_as_markdown(self, pdf_data: Dict, output_path: Path):
        """
        Lưu thành Markdown (đẹp hơn)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# {pdf_data['filename']}\n\n")
            f.write(f"**Trích xuất:** {pdf_data['extracted_at']}  \n")
            f.write(f"**Tổng số trang:** {pdf_data['total_pages']}  \n")
            
            if 'statistics' in pdf_data:
                stats = pdf_data['statistics']
                f.write(f"**Số ký tự:** {stats['total_characters']:,}  \n")
                f.write(f"**Số bảng:** {stats['total_tables']}  \n")
            
            f.write("\n---\n\n")
            
            # Content
            for page_data in pdf_data['pages']:
                f.write(f"## Trang {page_data['page']}\n\n")
                
                # Text
                f.write(page_data['text'])
                f.write("\n\n")
                
                # Tables
                if page_data['tables']:
                    for table in page_data['tables']:
                        f.write(f"### Bảng {table['table_id']}\n\n")
                        
                        rows = table['rows']
                        if not rows:
                            continue
                        
                        # Header
                        header = rows[0]
                        f.write("| " + " | ".join(header) + " |\n")
                        f.write("|" + "|".join([" --- " for _ in header]) + "|\n")
                        
                        # Data rows
                        for row in rows[1:]:
                            f.write("| " + " | ".join(row) + " |\n")
                        
                        f.write("\n")
                
                f.write("\n---\n\n")
        
        print(f"   💾 Đã lưu: {output_path.name}")
    
    def extract_all(self, formats: List[str] = ['txt', 'json', 'md']):
        """
        Trích xuất tất cả PDFs trong thư mục
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("⚠️  Không tìm thấy file PDF nào trong thư mục!")
            print(f"   Thư mục: {self.pdf_dir.absolute()}")
            return
        
        print(f"📚 Tìm thấy {len(pdf_files)} file PDF")
        print(f"📁 Output: {self.output_dir.absolute()}\n")
        
        # Process each PDF
        for pdf_path in pdf_files:
            # Extract
            pdf_data = self.extract_pdf(pdf_path)
            
            if 'error' in pdf_data:
                continue
            
            # Save in different formats
            base_name = pdf_path.stem
            
            if 'txt' in formats:
                txt_path = self.output_dir / f"{base_name}.txt"
                self.save_as_txt(pdf_data, txt_path)
            
            if 'json' in formats:
                json_path = self.output_dir / f"{base_name}.json"
                self.save_as_json(pdf_data, json_path)
            
            if 'md' in formats:
                md_path = self.output_dir / f"{base_name}.md"
                self.save_as_markdown(pdf_data, md_path)
            
            print()
        
        print(f"🎉 Hoàn tất! Kiểm tra thư mục: {self.output_dir}")
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """
        In thống kê tổng quan
        """
        print("\n" + "="*80)
        print("📊 TỔNG KẾT")
        print("="*80)
        
        # Count files
        txt_files = list(self.output_dir.glob("*.txt"))
        json_files = list(self.output_dir.glob("*.json"))
        md_files = list(self.output_dir.glob("*.md"))
        
        print(f"📄 TXT files:  {len(txt_files)}")
        print(f"📋 JSON files: {len(json_files)}")
        print(f"📝 MD files:   {len(md_files)}")
        
        # Total size
        total_size = sum(f.stat().st_size for f in self.output_dir.glob("*"))
        print(f"💾 Tổng dung lượng: {total_size / 1024 / 1024:.2f} MB")
        
        print("\n✅ Files đã sẵn sàng để sử dụng!")


# ========== MAIN ==========

if __name__ == "__main__":
    print("="*80)
    print("📄 PDF EXTRACTOR - Trích xuất nội dung PDF")
    print("="*80)
    print()
    
    # Khởi tạo extractor
    extractor = PDFExtractor(
        pdf_dir="./data/pdfs",
        output_dir="./data/extracted_content"
    )
    
    # Trích xuất tất cả
    # Formats: 'txt', 'json', 'md'
    extractor.extract_all(formats=['txt', 'json', 'md'])
    
    print("\n" + "="*80)
    print("🎯 Hướng dẫn sử dụng:")
    print("="*80)
    print("1. TXT:  Đọc trực tiếp, copy/paste")
    print("2. JSON: Load vào Python, xử lý structured data")
    print("3. MD:   Xem trên GitHub/VS Code, đẹp hơn")
    print()
    print("💡 Để dùng cho RAG: Chạy build_database.py với file TXT hoặc MD")