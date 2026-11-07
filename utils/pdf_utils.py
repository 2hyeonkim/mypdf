"""PDF 텍스트 추출 및 청크 분할 유틸리티."""

import io

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError, PyPdfError

# 페이지당 평균 글자 수가 이 값보다 적으면 스캔본(이미지 PDF)으로 간주
MIN_CHARS_PER_PAGE = 20


@st.cache_data(show_spinner=False, max_entries=50)
def extract_text(file_bytes: bytes, file_name: str) -> dict:
    """PDF 바이트에서 페이지별 텍스트를 추출한다.

    file_bytes 기준으로 캐싱되므로 같은 파일을 다시 업로드해도 재처리하지 않는다.
    파일은 메모리에서만 처리하며 디스크에 저장하지 않는다.

    Returns:
        {"name", "pages", "chars", "text", "error"}
        스캔본·암호화·손상 PDF는 text를 비우고 error에 사유를 담는다.
    """
    result = {"name": file_name, "pages": 0, "chars": 0, "text": "", "error": None}

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            # 열람 암호 없이 권한만 잠긴 PDF는 빈 비밀번호로 열 수 있음
            try:
                if not reader.decrypt(""):
                    result["error"] = "암호화된 PDF입니다. 암호를 해제한 뒤 다시 업로드해 주세요."
                    return result
            except (FileNotDecryptedError, PyPdfError, NotImplementedError):
                result["error"] = "암호화된 PDF입니다. 암호를 해제한 뒤 다시 업로드해 주세요."
                return result

        result["pages"] = len(reader.pages)
        page_texts = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""  # 특정 페이지 추출 실패는 건너뛰고 계속 진행
            if page_text.strip():
                page_texts.append(f"[페이지 {i}]\n{page_text.strip()}")
    except FileNotDecryptedError:
        result["error"] = "암호화된 PDF입니다. 암호를 해제한 뒤 다시 업로드해 주세요."
        return result
    except (PdfReadError, PyPdfError, ValueError, KeyError, OSError):
        result["error"] = "PDF 파일을 읽을 수 없습니다. 손상되었거나 올바른 PDF가 아닙니다."
        return result

    text = "\n\n".join(page_texts)
    result["text"] = text
    result["chars"] = len(text)

    if result["pages"] == 0:
        result["error"] = "페이지가 없는 PDF입니다."
    elif len(text.strip()) < MIN_CHARS_PER_PAGE * result["pages"] and len(text.strip()) < 100:
        result["text"] = ""
        result["chars"] = 0
        result["error"] = "텍스트를 추출할 수 없습니다 (스캔 이미지 PDF는 OCR 미지원)."

    return result


def split_into_chunks(text: str, chunk_size: int = 12000, overlap: int = 200) -> list[str]:
    """긴 텍스트를 chunk_size 글자 단위로 overlap만큼 겹치게 분할한다.

    가능하면 문단(빈 줄) 또는 줄바꿈 경계에서 자른다.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap은 0 이상, chunk_size 미만이어야 합니다.")
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 청크 뒤쪽 절반 안에서 가장 가까운 문단/줄 경계를 찾음
            boundary = text.rfind("\n\n", start + chunk_size // 2, end)
            if boundary == -1:
                boundary = text.rfind("\n", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
