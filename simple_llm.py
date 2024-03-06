import os
import gradio as gr
from langchain_community.llms import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="google/flan-ul2")

def chatbot(prompt):
    response = llm.invoke(prompt)
    return response  # Mengembalikan hasil teks dari model

demo = gr.Interface(fn=chatbot, inputs="text", outputs="text")
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
