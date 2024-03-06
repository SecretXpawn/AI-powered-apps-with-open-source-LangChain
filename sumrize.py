import os
import gradio as gr
from langchain_community.llms import HuggingFaceEndpoint
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import wget

# Mengatur token API untuk Hugging Face
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="google/flan-ul2")

# Mengakses ke dokumen
url = "https://raw.githubusercontent.com/Ichsan-Takwa/Generative-AI-Labs/main/Pembukaan_UUD_1945"
output_path = "pembukaanUUD1945.txt"  # Nama file lokal

# Mengecek jika file sudah ada
if not os.path.exists(output_path):
    # Mengunduh file menggunakan wget
    wget.download(url, out=output_path)

# Loader untuk teks
loader = TextLoader('pembukaanUUD1945.txt')

# Mengakses data
data = loader.load()

# Membuat instance untuk mencari data
index = VectorstoreIndexCreator().from_loaders([loader])

# Fungsi untuk merangkum query
def summarize(query):
    return index.query(query)

# Menjalankan Gradio
iface = gr.Interface(fn=summarize, inputs="text", outputs="text")
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)