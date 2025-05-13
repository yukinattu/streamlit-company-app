"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    if source.startswith("http"):
        return ct.LINK_SOURCE_ICON
    return ct.DOC_SOURCE_ICON


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # ✅ secretsからAPIキーを読み込む
    llm = ChatOpenAI(
        model_name=ct.MODEL,
        temperature=ct.TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # プロンプトテンプレート（ユーザー質問を独立文に）
    question_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # 回答プロンプトテンプレート（モードにより分岐）
    if st.session_state.mode == ct.ANSWER_MODE_1:
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", question_answer_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Retriever作成（履歴に基づく問い合わせ対応）
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # 回答Chain作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)

    # Retriever + Chain 統合
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 実行してLLM応答を取得
    llm_response = chain.invoke({
        "input": chat_message,
        "chat_history": st.session_state.chat_history
    })

    # 会話履歴に追加
    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message),
        llm_response["answer"]
    ])

    return llm_response
