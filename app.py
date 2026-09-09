"""
Simple web app: upload or record Hindi audio, get a real/fake verdict plus
an explanation of what drove it -- the mel spectrogram and a Grad-CAM
heatmap over it (see gradcam.py; hooks the actual trained detector.pt, not
any pretrained/reference weights).

Usage:
    python app.py
Then open http://127.0.0.1:7860
"""

import gradio as gr
from predict import load_model
from gradcam import explain
from preprocess import CONFIG

_model = load_model()  # loaded once at startup, reused across requests
_ANALYZED_SEC = CONFIG["target_duration_sec"]


def analyze(audio_path):
    if audio_path is None:
        return None, None, None
    _, _, probs, spectrogram_rgb, overlay_rgb = explain(audio_path, model=_model)
    return probs, spectrogram_rgb, overlay_rgb


with gr.Blocks(title="FakeWave") as demo:
    gr.Markdown(
        '<h1 style="text-align:center">FakeWave</h1>\n'
        '<h3 style="text-align:center">Fake voice detection with XAI</h3>'
    )
    gr.Markdown("Upload a voice recording to find if it's real or fake.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Hindi audio clip (upload or record)")

    with gr.Row():
        result_label = gr.Label(label="Result", num_top_classes=2)

    gr.Markdown(f"*Analyzed: first {_ANALYZED_SEC:.1f} seconds of audio*")

    with gr.Row():
        spectrogram_img = gr.Image(
            label="Mel spectrogram (the model's actual input representation)",
            height=420,
        )
        overlay_img = gr.Image(
            label="Grad-CAM: time-frequency regions that drove the verdict",
            height=420,
        )

    with gr.Row():
        submit_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.ClearButton([audio_input, result_label, spectrogram_img, overlay_img])

    submit_btn.click(
        fn=analyze,
        inputs=audio_input,
        outputs=[result_label, spectrogram_img, overlay_img],
    )
    audio_input.change(
        fn=analyze,
        inputs=audio_input,
        outputs=[result_label, spectrogram_img, overlay_img],
    )

if __name__ == "__main__":
    demo.launch()
