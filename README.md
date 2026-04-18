# RECITE

**Revising Eligibility Criteria Incorporating Textual Evidence**

A benchmark for evaluating LLMs on eligibility criteria revision using protocol amendment evidence from ClinicalTrials.gov.

## Quick Start

```bash
# Install
uv sync

# See available commands
uv run recite --help

# Run the full benchmark pipeline
uv run recite benchmark init-benchmark
```

## Prompts & Methodology

Detailed prompt text (model prompts, judge rubrics, and evaluation methodology) and additional supporting code will be publicly released within approximately 2 months as we complete a routine institutional IP process. The JSON structure in `config/benchmark_prompts.json` shows the expected schema, and full content will be restored in place once this process concludes. All other code, data, and evaluation infrastructure are fully available now.

**Reviewer / early access:** We are happy to share the complete prompts and methodology details with reviewers or interested researchers upon request. Please contact Russell Ro (russell.ro@ucsf.edu) or Reza Abbasi-Asl (reza.abbasi-asl@ucsf.edu).

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
