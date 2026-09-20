"""MMLU pilot: does a system-prompt persona change what the model can do?

Split by utility so each piece can be read, tested and changed on its own:

    dataset.py   load the frozen test set
    cells.py     the system-prompt conditions, baseline first
    prompts.py   how a question is put, how the judge is asked to read it
    client.py    provider calls and the retry policy
    judge.py     LLM-judge extraction of the chosen option
    run.py       orchestration, resume, JSONL output
    analyze.py   accuracy, refusal, and where they diverge
"""
