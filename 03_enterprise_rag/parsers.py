import os
import csv
import sqlite3
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class CorpusIndexer:
    def __init__(self, db_path="/app/corpus_index.db", vlm_context=None):
        self.db_path = db_path
        self.vlm_context = vlm_context
        # Create directory if local dev
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Creates the lightweight SQLite database schema using FTS5 for deterministic lexical search."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
                    file_path UNINDEXED,
                    content
                )
            ''')
            conn.commit()

    def _save_to_db(self, file_path: str, content: str):
        """Saves content efficiently into SQLite to avoid holding data in memory."""
        with sqlite3.connect(self.db_path) as conn:
            # FTS5 doesn't support UNIQUE constraints, so we delete first to handle updates cleanly
            conn.execute('DELETE FROM documents WHERE file_path = ?', (file_path,))
            conn.execute('''
                INSERT INTO documents (file_path, content)
                VALUES (?, ?)
            ''', (file_path, content))
            conn.commit()

    def search(self, query: str, top_k: int = 3) -> list:
        """Deterministic lexical search using FTS5 BM25-based ranking."""
        stop_words = {'what', 'is', 'the', 'of', 'in', 'for', 'which', 'a', 'an', 'to', 'on', 'at', 'does', 'did', 'do', 'where', 'when', 'how', 'are', 'it', 'this'}
        terms = []
        for word in query.replace('?', '').replace('.', '').replace(',', '').split():
            clean_word = word.strip()
            if clean_word.lower() not in stop_words and len(clean_word) > 1:
                # Escape quotes to prevent FTS5 syntax errors
                safe_word = clean_word.replace('"', '')
                terms.append(f'"{safe_word}"')
        
        match_expr = " OR ".join(terms) if terms else '""'
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute(
                    'SELECT file_path, content FROM documents WHERE documents MATCH ? ORDER BY rank LIMIT ?',
                    (match_expr, top_k)
                )
                return [{"file_path": row[0], "content": row[1]} for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"FTS5 Search error for query '{match_expr}': {e}")
                return []

    def index_file(self, file_path: str) -> bool:
        """
        Attempts to parse and index a single file deterministically.
        Strict Fault Tolerance: Catches ALL exceptions silently.
        Returns True if successful, False if skipped/failed.
        """
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        try:
            if ext in ['.txt', '.log', '.py']:
                content = self._parse_text(file_path)
            elif ext == '.csv':
                content = self._parse_csv(file_path)
            elif ext == '.xlsx':
                content = self._parse_xlsx(file_path)
            elif ext == '.docx':
                content = self._parse_docx(file_path)
            elif ext == '.pdf':
                content = self._parse_pdf(file_path)
            elif ext in ['.png', '.jpg', '.jpeg']:
                content = self._parse_image(file_path)
            else:
                logger.debug(f"Unsupported format {ext}, skipping {file_path}")
                return False
            
            if content and content.strip():
                self._save_to_db(file_path, content)
                # Strict memory rule: Delete variable explicitly 
                del content
                return True
            
            return False
            
        except Exception as e:
            # Silent failure to guarantee pipeline robustness against encrypted/corrupt files
            logger.warning(f"Error parsing {file_path}: {type(e).__name__} - {str(e)}")
            return False

    def _parse_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _parse_csv(self, file_path: str) -> str:
        rows = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)

    def _parse_xlsx(self, file_path: str) -> str:
        import openpyxl
        text = []
        # data_only=True ensures we get values, not formula expressions
        wb = openpyxl.load_workbook(file_path, data_only=True)
        for sheet in wb.worksheets:
            text.append(f"--- Sheet: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                row_vals = [str(cell) for cell in row if cell is not None]
                if row_vals:
                    text.append(", ".join(row_vals))
        wb.close()
        return "\n".join(text)

    def _parse_docx(self, file_path: str) -> str:
        import docx
        text = []
        doc = docx.Document(file_path)
        
        # 1. Extract paragraphs
        for p in doc.paragraphs:
            if p.text.strip():
                text.append(p.text.strip())
                
        # 2. Extract embedded tables
        for table in doc.tables:
            for row in table.rows:
                row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_vals:
                    text.append(" | ".join(row_vals))
                    
        return "\n".join(text)

    def _parse_pdf(self, file_path: str) -> str:
        import pdfplumber
        from pdfminer.pdfdocument import PDFPasswordIncorrect
        text = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except PDFPasswordIncorrect:
            logger.warning(f"Encrypted PDF skipped: {file_path}")
            return ""
        except Exception as e:
            logger.warning(f"Failed parsing PDF {file_path}: {e}")
            return ""
        return "\n".join(text)

    def _parse_image(self, file_path: str) -> str:
        if not self.vlm_context or not getattr(self.vlm_context, 'model', None):
            logger.warning(f"VLM context not available. Skipping image OCR for {file_path}")
            return ""
        
        try:
            # 1. Image validation
            with Image.open(file_path) as img:
                img.verify()
                
            # 2. VLM Inference Route
            return self.vlm_context.extract_text_from_image(file_path)
        except Exception as e:
            logger.warning(f"Failed image VLM processing for {file_path}: {e}")
            return ""
