import gradio as gr
import rag_core

# 启动时加载模型和数据
rag_core.init()

def chat(question, history):
    if not question.strip():
        return "", history
    result = rag_core.ask(question, top_k=3)
    answer = result["answer"]
    # 可选：把来源也显示出来
    sources = "\n\n".join([f"· {s['text'][:80]}..." for s in result["sources"]])
    full = f"{answer}\n\n---\n**参考来源：**\n{sources}"
    history = history + [
    {"role": "user", "content": question},
    {"role": "assistant", "content": full}
]
    return "", history

with gr.Blocks(title="高性价比人生指南 RAG") as demo:
    gr.Markdown("# 高性价比人生指南 · 智能问答")
    gr.Markdown("基于《高性价比人生指南》的 RAG 问答系统")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="输入你的问题，例如：人应该追求什么？", label="你的问题")
    clear = gr.Button("清空对话")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7861)