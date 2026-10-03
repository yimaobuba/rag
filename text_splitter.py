from semantic_text_splitter import TextSplitter
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

def extract_epub_text(epub_path: str) -> str:
    # 读取epub电子书
    book = epub.read_epub(epub_path)
    chapters = []
    # 遍历所有章节，提取纯文本
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        # 用BeautifulSoup去除HTML标签，保留纯文本内容
        soup = BeautifulSoup(item.get_content(), "html.parser")
        chapters.append(soup.get_text(separator="\n"))
    # 拼接所有章节为完整长文本
    return "\n\n".join(chapters)

# 替换成你的epub文件路径
epub_file_path = "D:/HowToLiveBetter/HowToLiveBetter.epub"
long_text = extract_epub_text(epub_file_path)

# 接下来就可以像之前一样切割文本
splitter = TextSplitter(capacity=1000)  # 按字符分割
chunks = splitter.chunks(long_text)

# 打印查看分割结果
for i, chunk in enumerate(chunks, 1):
    if(i==10):
     print(f"===== 第{i}块 =====\n{chunk}\n")
     break
