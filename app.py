"""PDF 분석기 - Streamlit 메인 앱."""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.llm_utils import (
    ANALYSIS_TYPES,
    LLMError,
    analyze,
    chat,
    get_client,
    truncate_context,
)
from utils.pdf_utils import extract_text

load_dotenv()  # 로컬 개발용 .env 로드 (Streamlit Cloud에서는 파일이 없으므로 무시됨)


def get_config(key: str, default: str | None = None) -> str | None:
    """설정값을 st.secrets(Streamlit Cloud) → 환경변수(.env) 순서로 조회한다."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # 로컬에 secrets.toml이 없으면 st.secrets 접근 시 예외가 발생할 수 있음
        pass
    value = os.getenv(key)
    return value if value else default


OPENAI_API_KEY = get_config("OPENAI_API_KEY")
OPENAI_MODEL = get_config("OPENAI_MODEL", "gpt-6.1-terra")
LANGUAGES = ["한국어", "English", "日本語", "中文"]
PREVIEW_CHARS = 1500
TAB_FILES, TAB_ANALYSIS, TAB_CHAT = "📁 업로드 파일", "📊 분석 결과", "💬 문서에 질문하기"

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1200px; }
    [data-testid="stSidebar"] h3 { font-size: 1rem; margin-bottom: 0.25rem; }
    /* 모델 응답의 제목(#, ##)이 파일명 제목보다 커지지 않도록 축소 */
    [class*="st-key-result_body"] h1 { font-size: 1.35rem; }
    [class*="st-key-result_body"] h2 { font-size: 1.2rem; }
    [class*="st-key-result_body"] h3 { font-size: 1.1rem; }
    [class*="st-key-result_body"] h4 { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 PDF 분석기")
st.caption("PDF를 업로드하면 AI가 요약·분석하고 질문에 답해드립니다")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY가 설정되지 않았습니다. "
        "로컬에서는 `.env` 파일에, Streamlit Cloud에서는 앱 설정의 Secrets에 추가해 주세요."
    )
    st.stop()

client = get_client(OPENAI_API_KEY)

# ---------------------------------------------------------------- 세션 상태
st.session_state.setdefault("file_signature", None)
st.session_state.setdefault("analysis", None)  # {"markdown": str, "meta": dict}
st.session_state.setdefault("messages", [])


def reset_results() -> None:
    st.session_state.analysis = None
    st.session_state.messages = []


def clear_chat() -> None:
    st.session_state.messages = []


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.subheader("① PDF 업로드")
    uploaded_files = st.file_uploader(
        "분석할 PDF 파일을 선택하세요 (여러 개 가능)",
        type=["pdf"],
        accept_multiple_files=True,
        help="파일당 최대 200MB. 업로드한 파일은 저장되지 않고 메모리에서만 처리됩니다.",
    )

    st.divider()
    st.subheader("② 분석 설정")
    analysis_type = st.radio("분석 유형", ANALYSIS_TYPES)
    custom_prompt = None
    if analysis_type == "사용자 정의":
        custom_prompt = st.text_area(
            "분석 지시문",
            placeholder="예: 이 문서에서 위험 요소와 대응 방안을 표로 정리해 줘",
        )
    scope = st.radio(
        "분석 범위",
        ["파일별 개별 분석", "전체 통합 분석"],
        help="여러 파일을 업로드했을 때 파일마다 따로 분석할지, 하나로 합쳐 분석할지 선택합니다.",
    )
    language = st.selectbox("응답 언어", LANGUAGES)

    st.divider()
    st.subheader("③ 대화")
    st.button("🗑️ 대화 초기화", on_click=clear_chat, use_container_width=True)

    st.caption(f"모델: `{OPENAI_MODEL}`")

# ---------------------------------------------------------------- 빈 화면: 사용 안내
if not uploaded_files:
    if st.session_state.file_signature is not None:
        st.session_state.file_signature = None
        reset_results()

    st.write("")
    steps = [
        ("1", "PDF 업로드", "왼쪽 사이드바에서 분석할 PDF 파일을 하나 이상 올려 주세요."),
        ("2", "분석 설정", "분석 유형(요약·핵심 포인트·키워드·사용자 정의), 범위, 응답 언어를 고르세요."),
        ("3", "분석하기", "'분석 결과' 탭에서 분석을 실행하고, '문서에 질문하기' 탭에서 궁금한 점을 물어보세요."),
    ]
    for col, (num, title, desc) in zip(st.columns(3), steps):
        with col.container(border=True):
            st.markdown(f"#### {num}. {title}")
            st.write(desc)
    st.info("텍스트가 포함된 PDF만 분석할 수 있습니다. 스캔 이미지 PDF(OCR)는 지원하지 않습니다.", icon="ℹ️")
    st.stop()

# ---------------------------------------------------------------- 파일 처리
# 업로드 파일 구성이 바뀌면 이전 분석 결과와 대화 초기화
signature = tuple((f.name, f.size) for f in uploaded_files)
if signature != st.session_state.file_signature:
    st.session_state.file_signature = signature
    reset_results()

with st.spinner("PDF에서 텍스트를 추출하는 중..."):
    docs = [extract_text(f.getvalue(), f.name) for f in uploaded_files]

valid_docs = [d for d in docs if not d["error"]]


def combined_text(documents: list[dict]) -> str:
    return "\n\n".join(f"=== 파일: {d['name']} ===\n{d['text']}" for d in documents)


# 일반 탭은 재실행 후에도 선택 상태가 유지된다. on_change="rerun"을 쓰면 탭 전환이
# 재실행을 일으켜 진행 중인 분석이 중단되므로 사용하지 않는다.
tab_files, tab_analysis, tab_chat = st.tabs([TAB_FILES, TAB_ANALYSIS, TAB_CHAT])

# ---------------------------------------------------------------- 탭 1: 업로드 파일
with tab_files:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("업로드 파일", f"{len(docs)}개", border=True)
    c2.metric("분석 가능", f"{len(valid_docs)}개", border=True)
    c3.metric("총 페이지", f"{sum(d['pages'] for d in valid_docs):,}", border=True)
    c4.metric("총 글자 수", f"{sum(d['chars'] for d in valid_docs):,}", border=True)

    for d in docs:
        with st.container(border=True):
            if d["error"]:
                st.markdown(f"**📕 {d['name']}**")
                st.warning(d["error"])
                continue
            st.markdown(f"**📄 {d['name']}**")
            st.caption(f"{d['pages']}페이지 · {d['chars']:,}자")
            with st.expander("텍스트 미리보기"):
                preview = d["text"][:PREVIEW_CHARS]
                if len(d["text"]) > PREVIEW_CHARS:
                    preview += "\n\n... (이하 생략)"
                st.text(preview)

if not valid_docs:
    for tab in (tab_analysis, tab_chat):
        tab.error("분석 가능한 PDF가 없습니다. 텍스트가 포함된 PDF를 업로드해 주세요.")
    st.stop()

# ---------------------------------------------------------------- 탭 2: 분석 결과
with tab_analysis:
    multi = len(valid_docs) > 1
    scope_label = scope if multi else "단일 파일"
    st.caption(f"분석 유형: **{analysis_type}** · 범위: **{scope_label}** · 응답 언어: **{language}**")

    run = st.button("🔍 분석하기", type="primary")
    result_box = st.container(border=True)

    if run:
        if analysis_type == "사용자 정의" and not (custom_prompt or "").strip():
            result_box.warning("사이드바에 사용자 정의 분석 지시문을 입력해 주세요.")
        else:
            if scope == "파일별 개별 분석" or not multi:
                targets = [(d["name"], d["text"]) for d in valid_docs]
            else:
                targets = [("전체 통합", combined_text(valid_docs))]

            sections = []
            try:
                with result_box:
                    for i, (title, text) in enumerate(targets):
                        st.markdown(f"### 📄 {title}")
                        progress = st.empty()

                        def on_progress(done: int, total: int, _p=progress) -> None:
                            _p.progress(done / total, text=f"긴 문서를 나누어 분석하는 중... ({done}/{total})")

                        with st.spinner(f"'{title}' 분석 중..."):
                            stream = analyze(
                                client, OPENAI_MODEL, text, analysis_type, language, custom_prompt, on_progress
                            )
                            result = st.container(key=f"result_body_{i}").write_stream(stream)
                        progress.empty()
                        sections.append((title, result))
            except LLMError as e:
                result_box.error(str(e))
            else:
                st.session_state.analysis = {
                    "sections": sections,
                    "markdown": "\n\n".join(f"## {t}\n\n{r}" for t, r in sections),
                    "meta": {
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "type": analysis_type,
                        "scope": scope_label,
                        "language": language,
                        "files": [d["name"] for d in valid_docs],
                        "model": OPENAI_MODEL,
                    },
                }
                st.success("분석이 완료되었습니다.")
    elif st.session_state.analysis:
        with result_box:
            for i, (title, result) in enumerate(st.session_state.analysis["sections"]):
                st.markdown(f"### 📄 {title}")
                st.container(key=f"result_body_{i}").markdown(result)
    else:
        result_box.caption("분석 유형을 선택하고 '분석하기' 버튼을 눌러 주세요. 결과가 여기에 표시됩니다.")

    if st.session_state.analysis:
        meta = st.session_state.analysis["meta"]
        header = (
            f"# PDF 분석 결과\n\n"
            f"- 생성 일시: {meta['created']}\n"
            f"- 분석 유형: {meta['type']} ({meta['scope']})\n"
            f"- 응답 언어: {meta['language']}\n"
            f"- 모델: {meta['model']}\n"
            f"- 파일: {', '.join(meta['files'])}\n\n---\n\n"
        )
        report = header + st.session_state.analysis["markdown"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        with st.container(horizontal=True):
            st.download_button(
                "⬇️ Markdown(.md) 다운로드",
                report,
                file_name=f"pdf_analysis_{stamp}.md",
                mime="text/markdown",
                on_click="ignore",
            )
            st.download_button(
                "⬇️ 텍스트(.txt) 다운로드",
                report,
                file_name=f"pdf_analysis_{stamp}.txt",
                mime="text/plain",
                on_click="ignore",
            )

# ---------------------------------------------------------------- 탭 3: 문서 Q&A
with tab_chat:
    context, truncated = truncate_context(combined_text(valid_docs))
    if truncated:
        st.caption("⚠️ 문서가 길어 앞부분 일부만 질문의 근거로 사용됩니다.")

    history_box = st.container()
    with history_box:
        if not st.session_state.messages:
            st.caption("업로드한 문서 내용에 대해 자유롭게 질문해 보세요. 예: \"이 문서의 결론은 뭐야?\"")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    question = st.chat_input("문서 내용에 대해 질문하세요")
    if question:
        with history_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(
                        chat(client, OPENAI_MODEL, context, st.session_state.messages, question, language)
                    )
                except LLMError as e:
                    st.error(str(e))
                else:
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
