from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 📌 テスト対象のファイルパス
txt_path = "data/MTG議事録/議事録ルール.txt"

# 1. テキストファイルを読み込み
loader = TextLoader(txt_path, encoding="utf-8")
docs = loader.load()

# 2. チャンク（分割）設定を適用
splitter = CharacterTextSplitter(
    chunk_size=300,        # 本番と同じサイズに合わせる
    chunk_overlap=20,
    separator="\n"
)
chunks = splitter.split_documents(docs)

# 3. チャンクごとの中身を表示
for i, chunk in enumerate(chunks):
    print(f"\n--- チャンク {i+1} ---")
    print(chunk.page_content)
