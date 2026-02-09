
import json
import sys

sys.path.insert(0, "../generation")

from retrieval.baseline_retriever import BaselineRetriever
from question_rewriter_old import QuestionRewriter

# Init
retriever = BaselineRetriever(events_path="icews_2014_train.txt")
rewriter = QuestionRewriter(retriever)

with open("mini_qa_devset.json", "r") as f:
    data = json.load(f)

print("Qestion rewriter test")


for idx, item in enumerate(data[:5], 1):
    q = item["question_implicit"]
    result = rewriter.rewrite(q)

    print(f"\n[{idx}] Original: {q}")
    print(f"    Signal: {result['signal_type']}")
    print(f"    Anchor: {result['anchor_phrase']}")
    print(f"    Entities: {result['anchor_entities']}")
    print(f"    Timestamp: {result['anchor_timestamp']}")
    print(f"    Rewritten: {result['rewritten']}")
    print(f"    Changed: {result['was_rewritten']}")