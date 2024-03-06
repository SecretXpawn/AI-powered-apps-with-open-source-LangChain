import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import HuggingFaceEndpoint
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL

# Mengatur token API untuk Hugging Face
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="google/flan-ul2")

# Memuat alat-alat
tools = load_tools(["llm-math", "requests_all", "human"], llm=llm)
tools.append(PythonREPLTool())

# Menginisialisasi agen
agent = initialize_agent(tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# Menggunakan agen untuk mengeksekusi perintah
result = agent.invoke("buatlah matplotlib sederhana yang menampilkan fungsi sin dan tampilkan plot nya")
print(result)
