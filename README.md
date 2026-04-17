# RECITE

**Revising Eligibility Criteria Incorporating Textual Evidence**

A benchmark for evaluating LLMs on eligibility criteria revision using protocol amendment evidence from ClinicalTrials.gov. 3,116 instances across 5,735 trials, with LLM-as-judge evaluation.

## Benchmark Results

| Model | Binary Equiv. | Ordinal (0-4) | ≥3 Rate | ≥4 Rate |
|-------|:---:|:---:|:---:|:---:|
| GPT-4o | 85.8% | 3.4±0.7 | 91.3% | 49.2% |
| GPT-4o-mini | 84.2% | 3.4±0.7 | 90.1% | 52.3% |
| Qwen2.5-72B | 82.1% | 3.3±0.7 | 90.3% | 44.9% |
| Gemma-2-9B | 84.7% | 3.3±0.7 | 86.9% | 45.0% |
| Qwen2.5-7B | 80.3% | 3.3±0.8 | 81.4% | 46.8% |
| Qwen3-32B | 81.5% | 3.2±0.7 | 90.1% | 34.4% |
| Llama-3.1-70B | 76.2% | 3.2±0.7 | 86.2% | 31.5% |
| Gemma-2-27B | 83.2% | 3.3±0.7 | 89.5% | 43.1% |
| Gemma-2-2B | 78.6% | 3.1±0.8 | 78.2% | 38.4% |
| Qwen2.5-3B | 79.1% | 3.2±0.8 | 80.5% | 40.2% |
| Qwen2.5-0.5B | 68.3% | 2.8±0.9 | 65.4% | 28.1% |
| DeepSeek-R1-7B | 79.8% | 3.2±0.8 | 82.1% | 41.3% |
| Mistral-7B | 77.4% | 3.1±0.8 | 79.8% | 37.6% |

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Citation

```bibtex
@article{ro2026recite,
  title  = {{RECITE}: Revising Eligibility Criteria Incorporating Textual Evidence},
  author = {Ro, Russell and Abbasi-Asl, Reza},
  year   = {2026}
}
```
