"""OpenAI 호출 및 프롬프트 템플릿 유틸리티."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

from utils.pdf_utils import split_into_chunks

# 한 번에 모델에 보낼 문서 최대 글자 수. 이보다 길면 Map-Reduce로 처리한다.
SINGLE_PASS_CHARS = 24000
CHUNK_SIZE = 12000
CHUNK_OVERLAP = 200
# 문서 Q&A에서 컨텍스트로 넣을 최대 글자 수
CHAT_CONTEXT_CHARS = 24000
# Reduce 결과가 여전히 길 때 재귀적으로 줄이는 최대 깊이
MAX_REDUCE_DEPTH = 3
# Map 단계 동시 API 호출 수 (요청 한도(429)를 고려해 작게 유지)
MAP_WORKERS = 4

ANALYSIS_TYPES = ["전체 요약", "핵심 포인트", "키워드·주제 추출", "사용자 정의"]

PROMPT_TEMPLATES = {
    "전체 요약": (
        "다음 문서의 전체 내용을 구조적으로 요약하세요. "
        "문서의 목적, 주요 내용, 결론을 포함하고 소제목을 활용해 읽기 쉽게 정리하세요."
    ),
    "핵심 포인트": (
        "다음 문서에서 가장 중요한 핵심 포인트를 불릿 목록으로 정리하세요. "
        "중요한 수치, 날짜, 조건이 있다면 빠짐없이 포함하세요."
    ),
    "키워드·주제 추출": (
        "다음 문서의 핵심 키워드 10~20개와 주요 주제를 추출하세요. "
        "각 키워드에는 문서에서 어떤 의미로 쓰였는지 한 줄 설명을 붙이세요."
    ),
}

SYSTEM_PROMPT = (
    "당신은 PDF 문서를 정확하게 분석하는 전문 분석가입니다. "
    "반드시 제공된 문서 내용에 근거해 답하고, 문서에 없는 내용은 추측하지 마세요. "
    "결과는 Markdown 형식으로 작성하세요."
)


class LLMError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 오류 메시지를 담은 예외."""


def _friendly_error(e: Exception, model: str) -> LLMError:
    """OpenAI 예외를 사용자 친화적인 한국어 메시지로 변환한다.

    원본 메시지에는 API 키 일부가 포함될 수 있으므로 그대로 노출하지 않는다.
    """
    if isinstance(e, openai.AuthenticationError):
        msg = "API 키 인증에 실패했습니다 (401). OPENAI_API_KEY 값을 확인해 주세요."
    elif isinstance(e, openai.PermissionDeniedError):
        msg = f"'{model}' 모델에 접근 권한이 없습니다 (403). 계정의 모델 사용 권한을 확인해 주세요."
    elif isinstance(e, openai.NotFoundError):
        msg = f"'{model}' 모델을 찾을 수 없습니다 (404). OPENAI_MODEL 설정을 확인해 주세요."
    elif isinstance(e, openai.RateLimitError):
        msg = "요청 한도를 초과했거나 크레딧이 부족합니다 (429). 잠시 후 다시 시도하거나 결제 상태를 확인해 주세요."
    elif isinstance(e, openai.APITimeoutError):
        msg = "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    elif isinstance(e, openai.APIConnectionError):
        msg = "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요."
    elif isinstance(e, openai.BadRequestError):
        msg = "요청이 거부되었습니다 (400). 문서가 너무 길거나 모델이 요청 형식을 지원하지 않을 수 있습니다."
    elif isinstance(e, openai.APIStatusError):
        msg = f"OpenAI API 오류가 발생했습니다 (HTTP {e.status_code}). 잠시 후 다시 시도해 주세요."
    else:
        msg = f"분석 중 알 수 없는 오류가 발생했습니다 ({type(e).__name__})."
    return LLMError(msg)


def get_client(api_key: str) -> OpenAI:
    """OpenAI 클라이언트를 생성한다."""
    return OpenAI(api_key=api_key, timeout=180, max_retries=2)


def _complete(client: OpenAI, model: str, messages: list[dict]) -> str:
    """스트리밍 없이 전체 응답을 받아 문자열로 반환한다."""
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
    except openai.OpenAIError as e:
        raise _friendly_error(e, model) from e
    return resp.choices[0].message.content or ""


def _stream(client: OpenAI, model: str, messages: list[dict]) -> Iterator[str]:
    """응답을 텍스트 조각 단위로 스트리밍한다."""
    try:
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except openai.OpenAIError as e:
        raise _friendly_error(e, model) from e


def _instruction(analysis_type: str, custom_prompt: str | None) -> str:
    if analysis_type == "사용자 정의":
        if not custom_prompt or not custom_prompt.strip():
            raise LLMError("사용자 정의 프롬프트를 입력해 주세요.")
        return custom_prompt.strip()
    if analysis_type not in PROMPT_TEMPLATES:
        raise LLMError(f"알 수 없는 분석 유형입니다: {analysis_type}")
    return PROMPT_TEMPLATES[analysis_type]


def _messages(instruction: str, text: str, language: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{instruction}\n응답은 반드시 {language}로 작성하세요.\n\n<문서>\n{text}\n</문서>",
        },
    ]


def _map(
    client: OpenAI,
    model: str,
    text: str,
    instruction: str,
    language: str,
    progress_callback: Callable[[int, int], None] | None,
) -> str:
    """텍스트를 청크로 나눠 각각 부분 분석한 뒤 하나로 이어 붙인다."""
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    total = len(chunks)

    def run(i: int, chunk: str) -> str:
        map_instruction = (
            f"아래는 긴 문서의 일부({i}/{total})입니다. 이후 전체 결과로 통합될 예정이니, "
            f"다음 지시에 필요한 정보를 빠짐없이 추출·정리하세요.\n지시: {instruction}"
        )
        return f"## 부분 {i}/{total}\n" + _complete(client, model, _messages(map_instruction, chunk, language))

    # 청크별 분석은 서로 독립적이므로 병렬로 호출한다.
    # progress_callback은 Streamlit 요소를 갱신하므로 메인 스레드(as_completed 루프)에서만 호출한다.
    partials: list[str | None] = [None] * total
    if progress_callback:
        progress_callback(0, total)
    with ThreadPoolExecutor(max_workers=MAP_WORKERS) as pool:
        futures = {pool.submit(run, i, chunk): i for i, chunk in enumerate(chunks, start=1)}
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                partials[futures[future] - 1] = future.result()
                if progress_callback:
                    progress_callback(done, total)
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return "\n\n".join(partials)


def analyze(
    client: OpenAI,
    model: str,
    text: str,
    analysis_type: str,
    language: str = "한국어",
    custom_prompt: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[str]:
    """문서 텍스트를 분석하고 결과를 스트리밍으로 반환한다.

    문서가 SINGLE_PASS_CHARS보다 길면 청크별 부분 분석(Map) 후
    최종 통합(Reduce)한다. 부분 분석 진행 상황은 progress_callback(완료 수, 전체 수)로 알린다.

    Raises:
        LLMError: 입력 오류 또는 API 오류 (사용자 표시용 한국어 메시지).
    """
    instruction = _instruction(analysis_type, custom_prompt)
    if not text.strip():
        raise LLMError("분석할 텍스트가 없습니다.")

    if len(text) <= SINGLE_PASS_CHARS:
        return _stream(client, model, _messages(instruction, text, language))

    # Map: 부분 분석 결과가 여전히 길면 한 번 더 줄인다
    combined = text
    for _ in range(MAX_REDUCE_DEPTH):
        combined = _map(client, model, combined, instruction, language, progress_callback)
        if len(combined) <= SINGLE_PASS_CHARS:
            break
    else:
        combined = combined[:SINGLE_PASS_CHARS]

    # Reduce: 부분 결과를 최종 결과로 통합 (스트리밍)
    reduce_instruction = (
        "아래는 긴 문서를 여러 부분으로 나누어 분석한 결과입니다. "
        "중복을 제거하고 하나의 일관된 최종 결과로 통합하세요. "
        f"'부분 1/N' 같은 표시는 최종 결과에 남기지 마세요.\n최종 지시: {instruction}"
    )
    return _stream(client, model, _messages(reduce_instruction, combined, language))


def truncate_context(text: str, max_chars: int = CHAT_CONTEXT_CHARS) -> tuple[str, bool]:
    """Q&A 컨텍스트를 최대 글자 수로 자른다. (잘린 텍스트, 잘림 여부)를 반환한다."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def chat(
    client: OpenAI,
    model: str,
    context: str,
    history: list[dict],
    question: str,
    language: str = "한국어",
) -> Iterator[str]:
    """문서 내용을 컨텍스트로 질문에 답하고 결과를 스트리밍으로 반환한다.

    Args:
        context: 문서 텍스트 (truncate_context로 잘린 것).
        history: 이전 대화 [{"role": "user"|"assistant", "content": str}, ...].
    """
    if not question.strip():
        raise LLMError("질문을 입력해 주세요.")
    system = (
        f"{SYSTEM_PROMPT}\n"
        f"아래 문서 내용만을 근거로 사용자의 질문에 {language}로 답하세요. "
        "문서에서 답을 찾을 수 없으면 '문서에서 해당 내용을 찾을 수 없습니다'라고 답하세요. "
        "가능하면 근거가 된 페이지 번호([페이지 N])를 함께 밝히세요.\n\n"
        f"<문서>\n{context}\n</문서>"
    )
    # 오래된 대화는 잘라 토큰 사용량을 제한
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    messages = [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}]
    return _stream(client, model, messages)
