"""Next-Token Probability Lab - Gradio app (deploy on Hugging Face Spaces)."""
import json
import os

import gradio as gr

import core

DEFAULT = "The capital of France is"


def tab_next(prompt, temp, k, p):
    return core.next_token_fig(prompt, temp, int(k), p)


def tab_generate(prompt, temp, k, p, seed, filt, fewshot, make_img):
    full_prompt = (core.FEW_SHOT + prompt) if fewshot else prompt
    out = core.generate(full_prompt, 40, temp, int(k), p, int(seed), filt)
    text = prompt + out
    return text, (core.text_to_image(text) if make_img else None)


def results_md():
    if not os.path.exists("results.json"):
        return "`results.json` not found. Run `python evaluate.py` and commit the file."
    r = json.load(open("results.json"))
    lines = ["### Factual error rate vs temperature (GPT-2 small, 20 facts)",
             "| Temperature | Error rate | Generations |", "|---|---|---|"]
    for t, v in r["factual"]["by_temperature"].items():
        lines.append(f"| {t} | {v['error_rate']:.1%} | {v['n']} |")
    zs, fs = r["prompting"]["zero_shot_error"], r["prompting"]["few_shot_error"]
    lines += ["", f"**Prompt technique (T=0.2):** zero-shot error {zs:.1%} vs few-shot error {fs:.1%}", "",
              "### Toxic-token filter (blocklist)", "| Filter | Generations with banned words |", "|---|---|",
              f"| off | {r['toxic']['off_rate']:.1%} |", f"| on | {r['toxic']['on_rate']:.1%} |", "",
              "### Character-level perplexity (WikiText-2 validation slice)",
              "| Model | Bits per char | Perplexity per char |", "|---|---|---|"]
    for name, v in r["perplexity"].items():
        lines.append(f"| {name} | {v['bpc']:.3f} | {v['ppl']:.2f} |")
    return "\n".join(lines)


with gr.Blocks(title="Next-Token Probability Lab") as demo:
    gr.Markdown("# Next-Token Probability Lab\nSee how GPT-2 picks each next word and how sampling settings change it.")
    with gr.Tab("1. Next-token probabilities"):
        p1 = gr.Textbox(DEFAULT, label="Prompt")
        t1 = gr.Slider(0.1, 2.0, 1.0, step=0.05, label="Temperature")
        k1 = gr.Slider(0, 200, 0, step=1, label="Top-k (0 = off)")
        pp1 = gr.Slider(0.05, 1.0, 1.0, step=0.05, label="Top-p")
        b1 = gr.Button("Plot")
        plot1, info1 = gr.Plot(), gr.Markdown()
        b1.click(tab_next, [p1, t1, k1, pp1], [plot1, info1])
    with gr.Tab("2. Temperature 0.2 vs 1.2"):
        p2 = gr.Textbox(DEFAULT, label="Prompt")
        b2 = gr.Button("Compare")
        plot2, out2 = gr.Plot(), gr.Markdown()
        b2.click(core.compare_temps_fig, [p2], [plot2, out2])
    with gr.Tab("3. Generate text + image"):
        p3 = gr.Textbox("Once upon a time in a quiet village", label="Prompt")
        with gr.Row():
            t3 = gr.Slider(0.1, 2.0, 0.8, step=0.05, label="Temperature")
            k3 = gr.Slider(0, 200, 50, step=1, label="Top-k")
            pp3 = gr.Slider(0.05, 1.0, 0.95, step=0.05, label="Top-p")
        with gr.Row():
            seed3 = gr.Number(0, label="Seed", precision=0)
            filt3 = gr.Checkbox(True, label="Toxic-token filter (guardrail)")
            few3 = gr.Checkbox(False, label="Prepend few-shot examples")
            img3 = gr.Checkbox(True, label="Make image from text")
        b3 = gr.Button("Generate")
        out3, image3 = gr.Textbox(label="Generated text", lines=6), gr.Image(label="Image")
        b3.click(tab_generate, [p3, t3, k3, pp3, seed3, filt3, few3, img3], [out3, image3])
    with gr.Tab("4. Self-attention"):
        p4 = gr.Textbox("The cat sat on the mat because it was tired", label="Prompt (max 40 tokens)")
        b4 = gr.Button("Show attention")
        plot4 = gr.Plot()
        b4.click(core.attention_fig, [p4], [plot4])
    with gr.Tab("5. BERT vs GPT-2"):
        gr.Markdown("BERT reads both sides of `[MASK]`; GPT-2 only sees the words before it.")
        p5 = gr.Textbox("The capital of France is [MASK].", label="Sentence with [MASK]")
        b5 = gr.Button("Predict")
        tbl5 = gr.Dataframe(headers=["BERT word", "BERT p", "GPT-2 word", "GPT-2 p"])
        b5.click(core.bert_vs_gpt, [p5], [tbl5])
    with gr.Tab("6. Measured results"):
        gr.Markdown(results_md())

if __name__ == "__main__":
    demo.launch()
