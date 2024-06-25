# SU4MT
Code for EMNLP 2023 paper "Enhancing Neural Machine Translation with Semantic Units"

# Usage
## WPE
Please refer to https://github.com/shrango/Words-Pair-Encoding for guidelines.

## SU4MT
We put example training and generating scripts in the `scripts` file. Following the instructions in the scripts to fill the path of processed data. Although the scripts are for En-Ro task, you can change the lang_ids as your wish.

# Citing

```
@inproceedings{huang-etal-2023-enhancing,
    title = "Enhancing Neural Machine Translation with Semantic Units",
    author = "Huang, Langlin  and
      Gu, Shuhao  and
      Zhuocheng, Zhang  and
      Feng, Yang",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.149",
    doi = "10.18653/v1/2023.findings-emnlp.149",
    pages = "2264--2277",
    abstract = "Conventional neural machine translation (NMT) models typically use subwords and words as the basic units for model input and comprehension. However, complete words and phrases composed of several tokens are often the fundamental units for expressing semantics, referred to as semantic units. To address this issue, we propose a method Semantic Units for Machine Translation (SU4MT) which models the integral meanings of semantic units within a sentence, and then leverages them to provide a new perspective for understanding the sentence. Specifically, we first propose Word Pair Encoding (WPE), a phrase extraction method to help identify the boundaries of semantic units. Next, we design an Attentive Semantic Fusion (ASF) layer to integrate the semantics of multiple subwords into a single vector: the semantic unit representation. Lastly, the semantic-unit-level sentence representation is concatenated to the token-level one, and they are combined as the input of encoder. Experimental results demonstrate that our method effectively models and leverages semantic-unit-level information and outperforms the strong baselines.",
}
```
