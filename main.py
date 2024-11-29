import gradio as gr
from Retriever import QueryEngineWrapper

query_engine = QueryEngineWrapper()


def chat_with_model(prompt, chat_history):
    chat_history.append(("User", prompt))
    response = query_engine.query(prompt)
    chat_history.append(("Bot", str(response)))
    return chat_history, ""


def on_slider_change(slider, value):
    return f"Slider {slider} value updated to: {value}"


def button_callback(button_label):
    return f"Button {button_label} clicked!"


with gr.Blocks() as demo:
    with gr.Accordion("Settings", open=False):
        with gr.Row():
            sliders = []
            for i in range(1, 6):
                slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label=f"Slider {i}",
                    interactive=True,
                )
                sliders.append(slider)
        slider_outputs = [
            gr.Textbox(label=f"Output for Slider {i+1}") for i in range(5)
        ]
        for slider, output in zip(sliders, slider_outputs):
            slider.change(on_slider_change, slider, output)

        with gr.Row():
            for label in ["Button 1", "Button 2", "Button 3"]:
                button = gr.Button(label)
                button.click(
                    button_callback,
                    inputs=[gr.Text(label=label)],
                    outputs=gr.Text(label=f"Output {label}")
                )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation")
            chat_input = gr.Textbox(
                placeholder="Enter your message here...",
                label="Input"
            )
            send_button = gr.Button("Send")

            chat_history = gr.State([])
            send_button.click(
                chat_with_model,
                inputs=[chat_input, chat_history],
                outputs=[chatbot, chat_input]
            )

demo.launch(server_name="0.0.0.0", server_port=2345)
