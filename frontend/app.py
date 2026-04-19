"""Gradio frontend scaffold."""

import gradio as gr


def ping() -> str:
    return "Frontend scaffold is ready"


demo = gr.Interface(fn=ping, inputs=None, outputs="text", title="DSA Solver")


if __name__ == "__main__":
    demo.launch()
