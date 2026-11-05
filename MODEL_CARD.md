# Model card: Next-Token Probability Lab

## Models used (not trained by this project, except the LSTM)
| Model | Use | Licence |
|---|---|---|
| gpt2 (124M) | next-token probabilities, generation, attention | Modified MIT |
| bert-base-uncased | masked-word comparison | Apache-2.0 |
| stabilityai/sd-turbo (hosted API, optional) | text to image | Stability AI community licence |
| Char-LSTM (trained in `evaluate.py`) | perplexity baseline | project code |

## Intended use
Teaching tool for understanding sampling in language models. Not for factual lookup, advice, moderation or any decision about people.

## Data
GPT-2: WebText. BERT: BooksCorpus + English Wikipedia. LSTM baseline: WikiText-2 slice (CC BY-SA 3.0).

## Measured performance
See `results.json` and Section 7 of the report. Factual error rate was measured on 20 prompts across temperatures; GPT-2 small is expected to be wrong often even at low temperature.

## Limitations and risks
- GPT-2 hallucinates and can produce biased, offensive or false text; more so at high temperature.
- The toxic-token filter is a short blocklist on single-token spellings: it misses misspellings, multi-token words, implicit toxicity and other languages.
- No demographic-bias measurement is included.
- Image output (if enabled) depends on a third-party model and may reflect its biases.

## Safety measures
Logit-level blocklist during generation; input/attention length caps; API key stored only as a Space secret.

## Contact
<Your name, roll no. 30, email>


## Evaluation status

No real evaluation has been run for this submission package. Any bundled numeric values are illustrative placeholders only and must not be interpreted as measured model performance.
