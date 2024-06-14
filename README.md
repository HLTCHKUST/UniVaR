# High-Dimension Human Value Representation in Large Language Models

This is the official reposity for ["High-Dimension Human Value Representation in Large Language Models" paper](https://arxiv.org/abs/2404.07900). 

In this paper we introduce UniVaR,a high-dimensional neural representation of symbolic human value distributions in LLMs. This is a continuous and scalable representation, self-supervised from the value-relevant responses of 8 LLMs and evaluated on 15 open-source and commercial LLMs.

### File Structure
- univar_trainer => Folder containing training scripts for building UniVar models.
- univar_evaluation => Folder containing the evaluation scripts for evaluating UniVar and other representation models.
- univar_inference_demo =>
- value_eliciting_qa_generation => Folder containing pipeline generation scripts for value eliciting QAs.

### Citation
```
@misc{cahyawijaya2024highdimension,
      title={High-Dimension Human Value Representation in Large Language Models}, 
      author={Samuel Cahyawijaya and Delong Chen and Yejin Bang and Leila Khalatbari and Bryan Wilie and Ziwei Ji and Etsuko Ishii and Pascale Fung},
      year={2024},
      eprint={2404.07900},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```