import gradio as gr
from predict import predict_sentiment

def analyze(text):
    return predict_sentiment(text)

interface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=3, placeholder="Enter text..."),
    outputs="text",
    title="Sentiment Model"
)

interface.launch()