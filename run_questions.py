import time

import config
import rag

# one question per line in questions.txt
lines = (config.BASE_DIR / "questions.txt").read_text(encoding="utf-8").splitlines()
questions = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

output = ["# Results", ""]
for number, question in enumerate(questions, start=1):
    print(f"[{number}/{len(questions)}] {question}")
    result = rag.answer_question(question)
    output += [f"## {number}. {question}", "", result["answer"], "", "Sources: " + "; ".join(result["sources"]), ""]
    time.sleep(2)

(config.BASE_DIR / "results.md").write_text("\n".join(output), encoding="utf-8")
print("saved results.md")
