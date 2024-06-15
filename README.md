# High-Dimension Human Value Representation in Large Language Models

<img width="836" alt="cultural_map" src="https://github.com/HLTCHKUST/UniVaR/assets/2826602/6e733133-a525-417b-b6c4-cded996a325d">

This is the official reposity for ["High-Dimension Human Value Representation in Large Language Models" paper](https://arxiv.org/abs/2404.07900). 

In this paper we introduce UniVaR,a high-dimensional neural representation of symbolic human value distributions in LLMs. This is a continuous and scalable representation, self-supervised from the value-relevant responses of 8 LLMs and evaluated on 15 open-source and commercial LLMs.

## What is UniVaR?

<img width="894" alt="Screenshot 2024-06-15 at 1 05 31 PM" src="https://github.com/HLTCHKUST/UniVaR/assets/2826602/85c51f5d-6f65-40c9-b68d-7a1884e2c736">

### Value Eliciting QA
<img width="879" alt="Screenshot 2024-06-15 at 1 05 42 PM" src="https://github.com/HLTCHKUST/UniVaR/assets/2826602/33d369c3-fc59-4c8d-bec8-d43ead9e1501">


## UniVaR is not a Sentence Embedding
<img width="885" alt="Screenshot 2024-06-15 at 1 06 13 PM" src="https://github.com/HLTCHKUST/UniVaR/assets/2826602/f7a2fa81-322e-4845-b784-ce229adbd161">

<img width="884" alt="Screenshot 2024-06-15 at 1 05 57 PM" src="https://github.com/HLTCHKUST/UniVaR/assets/2826602/0b23b35f-1ad6-4424-b188-01ec282a2850">

## File Structure
- value_eliciting_qa_generation => Folder containing pipeline generation scripts for value eliciting QAs.
- univar_trainer => Folder containing training scripts for building UniVar models.
- univar_evaluation => Folder containing the evaluation scripts for evaluating UniVar and other representation models.
- univar_inference_demo =>

## Citation
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
