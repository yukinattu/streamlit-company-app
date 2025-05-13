import streamlit as st
import utils
import constants as ct

# ファイルパスにページ番号を追加する共通関数
def format_with_page_info(path: str, page: int = None) -> str:
    if path.lower().endswith(".pdf") and page is not None:
        return f"{path}（ページNo.{page + 1}）"
    return path

def display_app_title():
    st.markdown(f"<h1 style='text-align: center;'>{ct.APP_NAME}</h1>", unsafe_allow_html=True)
    st.success("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。\nサイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")
    st.warning("具体的に入力したほうが期待通りの回答を得やすいです。")

def display_select_mode():
    st.sidebar.title("利用目的")
    st.sidebar.radio("", [ct.ANSWER_MODE_1, ct.ANSWER_MODE_2], key="mode")

def display_initial_ai_message():
    with st.sidebar:
        st.markdown("**【「社内文書検索」を選択した場合】**")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

        st.markdown("**【「社内問い合わせ」を選択した場合】**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)

def display_conversation_log():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message["content"]["mode"] == ct.ANSWER_MODE_1:
                    if not "no_file_path_flg" in message["content"]:
                        st.markdown(message["content"]["main_message"])
                        icon = utils.get_source_icon(message['content']['main_file_path'])
                        st.success(format_with_page_info(message['content']['main_file_path'], message['content'].get('main_page_number')), icon=icon)

                        if "sub_message" in message["content"]:
                            st.markdown(message["content"]["sub_message"])
                            for sub_choice in message["content"]["sub_choices"]:
                                icon = utils.get_source_icon(sub_choice['source'])
                                st.info(format_with_page_info(sub_choice['source'], sub_choice.get('page_number')), icon=icon)
                    else:
                        st.markdown(message["content"]["answer"])
                else:
                    st.markdown(message["content"]["answer"])
                    if "file_info_list" in message["content"]:
                        st.divider()
                        st.markdown(f"##### {message['content']['message']}")
                        for file_info in message["content"]["file_info_list"]:
                            icon = utils.get_source_icon(file_info)
                            st.info(file_info, icon=icon)

def display_search_llm_response(llm_response):
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:
        main_file_path = llm_response["context"][0].metadata["source"]
        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)
        icon = utils.get_source_icon(main_file_path)

        if "page" in llm_response["context"][0].metadata:
            main_page_number = llm_response["context"][0].metadata["page"]
            st.success(format_with_page_info(main_file_path, main_page_number), icon=icon)
        else:
            st.success(main_file_path, icon=icon)

        sub_choices = []
        duplicate_check_list = []

        for document in llm_response["context"][1:]:
            sub_file_path = document.metadata["source"]
            if sub_file_path == main_file_path or sub_file_path in duplicate_check_list:
                continue
            duplicate_check_list.append(sub_file_path)
            if "page" in document.metadata:
                sub_choice = {"source": sub_file_path, "page_number": document.metadata["page"]}
            else:
                sub_choice = {"source": sub_file_path}
            sub_choices.append(sub_choice)

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)
            for sub_choice in sub_choices:
                icon = utils.get_source_icon(sub_choice['source'])
                st.info(format_with_page_info(sub_choice['source'], sub_choice.get('page_number')), icon=icon)

        content = {
            "mode": ct.ANSWER_MODE_1,
            "main_message": main_message,
            "main_file_path": main_file_path,
        }
        if "page" in llm_response["context"][0].metadata:
            content["main_page_number"] = main_page_number
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices
    else:
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content = {
            "mode": ct.ANSWER_MODE_1,
            "answer": ct.NO_DOC_MATCH_MESSAGE,
            "no_file_path_flg": True
        }
    return content

def display_contact_llm_response(llm_response):
    st.markdown(llm_response["answer"])
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()
        message = "情報源"
        st.markdown(f"##### {message}")
        file_path_list = []
        file_info_list = []

        for document in llm_response["context"]:
            file_path = document.metadata["source"]
            if file_path in file_path_list:
                continue

            if "page" in document.metadata:
                page_number = document.metadata["page"]
                file_info = format_with_page_info(file_path, page_number)
            else:
                file_info = file_path

            icon = utils.get_source_icon(file_path)
            st.info(file_info, icon=icon)

            file_path_list.append(file_path)
            file_info_list.append(file_info)

    content = {
        "mode": ct.ANSWER_MODE_2,
        "answer": llm_response["answer"]
    }
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = message
        content["file_info_list"] = file_info_list

    return content
