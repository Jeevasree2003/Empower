import argparse
import json

from metrics import bleu_metric, f1_metric
from rouge_score import rouge_scorer

NO_PASSAGE_USED = "no_passages_used"
KNOWLEDGE_SEP = "__knowledge__"


def rouge_l_score(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file")
    parser.add_argument("--test_data")
    parser.add_argument("--eval_metric", default="kf1", choices=["f1", "kf1", "rouge_l", "bleu"])
    args = parser.parse_args()

    hyp_list = []
    ref_list = []
    knowledge_list = []

    with open(args.pred_file, mode="r", encoding="utf-8") as rf:
        for line in rf.readlines():
            _line = line.split("|||")
            assert len(_line) == 2
            hyp_list.append(_line[0].strip())
            ref_list.append(_line[1].strip())

    with open(args.test_data, mode="r", encoding="utf-8") as rf_test:
        for line in rf_test:
            record = json.loads(line)
            knowledge_field = record["knowledge"][0]
            if KNOWLEDGE_SEP in knowledge_field:
                knowledge = knowledge_field.split(KNOWLEDGE_SEP, 1)[1].strip()
            else:
                knowledge = knowledge_field.strip()
            knowledge_list.append(knowledge)

    if args.eval_metric == "kf1":
        hyp_rm_no_pass_used = []
        kno_rm_no_pass_used = []
        assert len(hyp_list) == len(knowledge_list)
        for hyp, know in zip(hyp_list, knowledge_list):
            if know != NO_PASSAGE_USED:
                hyp_rm_no_pass_used.append(hyp)
                kno_rm_no_pass_used.append(know)

        print(f"KF1: {f1_metric(hyp_rm_no_pass_used, kno_rm_no_pass_used)}")
    else:
        assert len(hyp_list) == len(ref_list)
        if args.eval_metric == "f1":
            print(f"F1: {f1_metric(hyp_list, ref_list)}")
        elif args.eval_metric == "rouge_l":
            rl = sum(rouge_l_score(hyp, ref) for hyp, ref in zip(hyp_list, ref_list)) / len(hyp_list)
            print(f"rouge-l: {rl}")
        elif args.eval_metric == "bleu":
            b1, b2, b3, b4 = bleu_metric(hyp_list, ref_list)
            print(f"bleu-1: {b1}")
            print(f"bleu-2: {b2}")
            print(f"bleu-3: {b3}")
            print(f"bleu-4: {b4}")
        else:
            raise ValueError(f"Unsupported metric: {args.eval_metric}")


if __name__ == "__main__":
    main()
