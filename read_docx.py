#!/usr/bin/env python3
"""
Script para leer documentos .docx y convertirlos a texto plano
"""

import sys
from docx import Document

def read_docx_to_text(docx_path):
    """Lee un archivo .docx y retorna el texto completo"""
    try:
        doc = Document(docx_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        
        return '\n'.join(full_text)
    
    except Exception as e:
        print(f"Error reading docx: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_docx.py <docx_file>")
        sys.exit(1)
    
    docx_file = sys.argv[1]
    text = read_docx_to_text(docx_file)
    
    if text:
        print(text)
    else:
        print("Failed to read document")
        sys.exit(1)