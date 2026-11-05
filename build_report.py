"""Build the PDF report. Reads results.json if it exists (run evaluate.py first).

    python build_report.py --name "Your Name" --roll 30 --url https://huggingface.co/spaces/USER/P5 --repo https://github.com/USER/P5
Output: <roll>_<name>_P5.pdf
"""
import argparse
import json
import os

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Your_Name")
ap.add_argument("--roll", default="30")
ap.add_argument("--url", default="[LIVE APP URL]")
ap.add_argument("--repo", default="[GITHUB URL]")
a = ap.parse_args()

R = json.load(open("results.json")) if os.path.exists("results.json") else None
ss = getSampleStyleSheet()
H1, H2, B = ss["Heading1"], ss["Heading2"], ss["BodyText"]
NOTE = "<font color='#b91c1c'><b>[Run evaluate.py, then rebuild this report to fill this in]</b></font>"
S = []


def P(t, st=B):
    S.append(Paragraph(t, st))
    S.append(Spacer(1, 6))


def table(rows, widths=None):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    S.append(t)
    S.append(Spacer(1, 10))


def lifecycle():
    steps = ["Problem", "Data /\nPrompts", "Model\n(GPT-2,BERT)", "Sampling\nlayer", "Evaluate", "Deploy\n(Space)", "Monitor"]
    d = Drawing(480, 60)
    w = 60
    for i, s in enumerate(steps):
        x = i * 68
        d.add(Rect(x, 15, w, 32, fillColor=colors.HexColor("#e2e8f0"), strokeColor=colors.HexColor("#334155")))
        for j, line in enumerate(s.split("\n")):
            d.add(String(x + w / 2, 36 - j * 10, line, textAnchor="middle", fontSize=7))
        if i < len(steps) - 1:
            d.add(Line(x + w, 31, x + 68, 31, strokeColor=colors.HexColor("#334155")))
    return d


# ------------------------------------------------------------------ title
P("Next-Token Probability Lab (P5)", ss["Title"])
if R and "SYNTHETIC_PLACEHOLDER" in R.get("data_status",""):
    P("<b><font color='#b91c1c'>ILLUSTRATIVE PLACEHOLDER DATA — NOT MEASURED. Evaluation has not been run. Replace these values with genuine results before submission.</font></b>")
P(f"Generative AI - Individual CA Project, LY M.Sc. AI &amp; ML<br/>Name: {a.name} &nbsp; Roll No.: {a.roll}<br/>"
  f"Live app: {a.url}<br/>GitHub: {a.repo}")
P("1. Goal and scope", H1)
P("The lab shows how a language model picks each next word and how temperature, top-k and top-p change the result. "
  "The core feature is an interactive probability chart for GPT-2 small (124M parameters) with sampling implemented "
  "by hand in PyTorch. Around it sit a self-attention view, a BERT vs GPT-2 comparison, a text-to-image step, "
  "a toxic-token filter and a measured evaluation.")

P("2. Layer I - Foundations (Unit 1)", H1)
S.append(lifecycle())
P("<b>Lifecycle:</b> problem definition, prompts/data, pretrained model, sampling layer, evaluation, deployment, monitoring.")
P("<b>Generative vs discriminative.</b> A discriminative model learns p(y|x), for example BERT classifying a sentence. "
  "A generative language model learns p(x) = product of p(token<sub>t</sub> | tokens before t). GPT-2 is generative: at every step "
  "it outputs a probability for each of 50,257 tokens, and the lab plots exactly that distribution (the probability view).")
P("<b>Prompt techniques compared:</b> zero-shot vs few-shot prompting on 20 factual prompts (Section 7).")

P("3. Layer II - GANs and LLMs (Unit 2)", H1)
P("GPT-2 is a decoder-only Transformer. The Attention tab shows the last-layer self-attention matrix averaged over the 12 heads; "
  "the empty upper triangle is the causal mask, which stops a token from seeing the future. "
  "Prompt-based learning (few-shot prefix) is compared with plain zero-shot prompting. "
  "Relation to GANs: a GAN trains a generator against a discriminator; a language model has no discriminator and is trained "
  "only to maximise next-token likelihood, so the quality knobs are at sampling time (temperature, top-k, top-p).")

P("4. Layer III - GPT, BERT and multimodal (Unit 3)", H1)
P("GPT-2 (left-to-right) generates text. BERT-base-uncased (bidirectional masked LM) fills a [MASK]. "
  "The BERT vs GPT-2 tab shows both predictions for the same sentence: BERT sees words on both sides of the blank, "
  "GPT-2 only the words before it. The image modality is text to image (sd-turbo through the hosted Inference API; "
  "a plain text-card image is the offline fallback).")

P("5. Layer IV - Tools and generation (Unit 4)", H1)
P("Built in Python with PyTorch and Hugging Face Transformers; UI in Gradio; hosted on Hugging Face Spaces. "
  "<b>adjust()</b> in core.py implements temperature scaling, top-k and top-p (nucleus) filtering from scratch and "
  "<b>generate()</b> samples with torch.multinomial. Image generation: sd-turbo.")
table([["Step", "Operation"],
       ["1", "logits = GPT-2(prompt)[last position]"],
       ["2", "logits / temperature"],
       ["3", "keep only the k largest logits (top-k)"],
       ["4", "keep the smallest set whose probability sum reaches p (top-p)"],
       ["5", "softmax, then torch.multinomial draws one token; repeat"]], [1.5 * cm, 14 * cm])

P("6. Experimental setup", H1)
if R:
    st = R["settings"]
    P(f"Model: {st['model']}. {st['seeds_per_prompt']} random seeds per prompt. Run time {st.get('minutes', '?')} min. "
      f"{'<b>QUICK MODE (smoke test only)</b>' if st.get('quick') else ''}")
else:
    P(NOTE)
P("Factual test: 20 sentence-starters with a known one-word answer (e.g. 'The capital of France is'). A generation of 8 tokens "
  "counts as an error if the answer word is absent. Perplexity: a character-level LSTM trained here (WikiText-2 slice) against "
  "pretrained GPT-2, both scored in bits per character on the same validation slice.")

P("7. Results", H1)
if R:
    P("<b>7.1 Factual error rate vs temperature</b>")
    table([["Temperature", "Error rate", "Generations"]] +
          [[t, f"{v['error_rate']:.1%}", str(v["n"])] for t, v in R["factual"]["by_temperature"].items()])
    P("<b>7.2 Prompt technique (T = 0.2)</b>")
    table([["Technique", "Error rate"], ["Zero-shot", f"{R['prompting']['zero_shot_error']:.1%}"],
           ["Few-shot (3 examples)", f"{R['prompting']['few_shot_error']:.1%}"]])
    P("<b>7.3 Perplexity</b>")
    table([["Model", "Bits/char", "Perplexity/char"]] +
          [[n, f"{v['bpc']:.3f}", f"{v['ppl']:.2f}"] for n, v in R["perplexity"].items()])
    P("<b>7.4 Toxic-token filter</b>")
    t = R["toxic"]
    table([["Filter", "Generations containing a banned word"], ["Off", f"{t['off_rate']:.1%} (n={t['off_n']})"],
           ["On", f"{t['on_rate']:.1%} (n={t['on_n']})"]])
else:
    P(NOTE)
P("<i>Interpretation: write 3-4 sentences here after you see your own numbers - does error rise with temperature? "
  "does few-shot help? which model has lower bits per character and why?</i>")

P("8. Failure cases", H1)
if R:
    rows = [["Temp", "Prompt", "Expected", "GPT-2 produced"]]
    for tmp, items in R["factual"]["failure_cases"].items():
        for it in items[:3]:
            rows.append([tmp, Paragraph(it["prompt"], B), it["expected"], Paragraph(repr(it["got"]), B)])
    table(rows, [1.5 * cm, 6 * cm, 2.5 * cm, 6.5 * cm])
else:
    P(NOTE)
P("Known limits: GPT-2 small stores little factual knowledge, so errors occur even at low temperature; the substring check "
  "can miss correct paraphrases; the blocklist only bans single-token spellings.")

P("9. Layer V - Responsible AI (Unit 5)", H1)
P("<b>Measured test:</b> hallucination (factual error) rate vs temperature, Section 7.1. "
  "<b>Guardrail:</b> logit-level toxic-token blocklist during generation, Section 7.4, plus an input length cap. "
  "<b>Bias note:</b> GPT-2 was trained on web text (WebText, outbound Reddit links) and reflects its stereotypes; "
  "this lab does not measure demographic bias and says so. "
  "<b>Licence / IP:</b> GPT-2 - modified MIT licence; bert-base-uncased - Apache-2.0; sd-turbo - Stability AI community licence "
  "(check terms for commercial use); WikiText-2 - CC BY-SA 3.0. Generated text and images are not guaranteed free of copyrighted material. "
  "<b>Model card:</b> MODEL_CARD.md in the repository.")

P("10. Conclusion", H1)
P("This project demonstrates how temperature reshapes the next-token distribution, the trade-off between diversity and factual "
  "accuracy, and what you would improve (larger model, better toxicity classifier, bias probes).")
P("References", H2)
P("Radford et al. (2019) Language Models are Unsupervised Multitask Learners (GPT-2). Devlin et al. (2019) BERT. "
  "Vaswani et al. (2017) Attention Is All You Need. Holtzman et al. (2020) The Curious Case of Neural Text Degeneration (top-p).")

out = f"{a.roll}_{a.name.replace(' ', '_')}_P5.pdf"
SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm).build(S)
print("wrote", out)
