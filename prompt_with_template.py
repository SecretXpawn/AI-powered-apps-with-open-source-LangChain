import os
import gradio as gr
from langchain_community.llms import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="google/flan-ul2")

def generate_cover_letter(posisi: str, perusahaan: str, keterampilan: str) -> str:
    prompt = (
        f"Yang terhormat HRD Manajer {perusahaan},\n\n"
        f"Dengan surat ini, saya [NAMA KAMU], ingin melamar untuk posisi {posisi} di {perusahaan}. "
        f"Saya memiliki pengalaman di bidang {keterampilan}. Terima kasih telah mempertimbangkan lamaran saya.\n\n"
        "Hormat saya,\n[NAMA KAMU]"
    )
    response = llm.invoke(prompt)
    return response

# Define the Gradio interface inputs
inputs = [
    gr.Textbox(label="Posisi"),
    gr.Textbox(label="Perusahaan"),
    gr.Textbox(label="Keterampilan")
]

# Define the Gradio interface output
output = gr.Textbox(label="Template Surat")

# Launch the Gradio interface
gr.Interface(fn=generate_cover_letter, inputs=inputs, outputs=output).launch(server_name="0.0.0.0", server_port=7860, share=True)
