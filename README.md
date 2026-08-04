## Requirments ##

This code is tested on Python 3.8, Pytorch 1.7.1, and transformers 4.18.0

`pip install -r requirements.txt`

`python -m spacy download en_core_web_sm`

See [PIPELINE.md](PIPELINE.md) for the full end-to-end pipeline (KTC preprocessing, dataset splits, training, inference, evaluation).

---

## Training ##

The EMPOWER model is trained in two levels. It is dual-tier deep prompt tuning approach. In the first level, it encapsulates the relevant knowledge into continuous knowledge-attributed deep prompts and in second level, it utilizes the encoded knowledge for knowledge-aware response generation.


**Stage 1: "KDPT: Knowledge-attributed Deep Prompt Tuning"**
For the first level training, run kdpt.sh script file.


**Stage 2: "RDPT: Response-attributed Deep Prompt Tuning"**
For RDPT tarining, run rdpt.sh script file.

---

## Inference ##
Once the training is finished, run the gen.sh script file to conduct the inference stage.





