import os
import gradio as gr
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory, ConversationSummaryMemory
import time

# Mengatur token API untuk Hugging Face
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="google/flan-ul2")

# Mengatur memori percakapan
conv_memory = ConversationBufferWindowMemory(
    memory_key="chat_history_lines",
    input_key="input",
    k=1
)
summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
memory = CombinedMemory(memories=[conv_memory, summary_memory])

# Mendefinisikan template untuk prompt percakapan
_DEFAULT_TEMPLATE = """Berikut percakapan persahabatan antara manusia dan AI. AI ini banyak bicara dan memberikan banyak detail spesifik dari konteksnya. Jika AI tidak mengetahui jawaban atas sebuah pertanyaan, AI dengan jujur mengatakan bahwa ia tidak mengetahuinya.
Ringkasan percakapan:
{history}
Percakapan saat ini:
{chat_history_lines}
Manusia: {input}
AI:"""

# Create the prompt from the template
PROMPT = PromptTemplate(
    input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE
)

# Set up the conversation chain
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    prompt=PROMPT
)

# Set up the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()  # The chatbot object
    msg = gr.Textbox(label="pesan")  # The textbox for user input
    clear = gr.Button("Clear")  # The button to clear the chatbox

    # Define the function that will handle user input and generate the bot's response
    def respond(message, chat_history):
        bot_message = conversation.run(message)  # Run the user's message through the conversation chain
        chat_history.append((message, bot_message))  # Append the user's message and the bot's response to the chat history
        time.sleep(1)  # Pause for a moment
        return "", chat_history  # Return the updated chat history

    # Connect the respond function to the textbox and chatbot
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # Connect the "Clear" button to the chatbot
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
