# UniVaR Value Eliciting QA Generation Scripts

### Generation Pipeline
1. We generate the value eliciting question given the situation from the `data/*/*_situation.py` with `generate_prompt.py`
2. The generated value eliciting questions are then translated into multiple languages using NLLB-200 with `mt2oth.py`
3. The translated value eliciting questions are fed to LLMs to generate the answers either through:
  a. generation API with `gen_answer_llm_api.py`, or
  b. self-hosted LLM with `gen_answer_llm.py`
4. The generated QAs are translated back to English using NLLB-200 with `mt2eng.py`

**Notes**: Paraphrasing can be done by translating the questions before step (2) or after step (4) to ensure the quality of the paraphrasing as LLMs tend to perform best on English.