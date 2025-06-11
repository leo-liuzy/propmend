from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd
from glob import glob
from bespokelabs import curator
from datasets import Dataset

import re
def tag_content_extractor(tag):
    pattern = rf"<{tag}>([\s\S]*?)(?:</{tag}>|$)"
    def content_extractor(text):
        return re.findall(pattern, text)
    return content_extractor

score_tag_extractor = tag_content_extractor("score")


class LlmAsJudge(curator.LLM):
    MAX_VAL: float = 10.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy: 
                Score 0: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Ground truth]
{reference}
[The End of Ground truth]

[The Start of Assistant's Answer]
{prediction}
[The End of Assistant's Answer]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"], reference=input["answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= 10
        score = float(score)
        score /= self.MAX_VAL
        if "llm_accuracy" in input:
            existing_llm_acc = input["llm_accuracy"]
            del input["llm_accuracy"]
        input["llm_accuracy"] = score

        return {**input}


llm_judge = LlmAsJudge(
    model_name="gpt-4o-mini", backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
)

for fpath in tqdm(
    [
        "/data/users/zliu/mend/controlled_ripple_exp_output/qwen_share_max_4K_14_27/controlled_ripple_edit/mend_eval_loss=clm_input=seen_n=500_prompt=no_w-gen_wo-icl_4K_test_id-question.xlsx",
    ]
):
    scored_df = pd.read_excel(fpath)

    scored_df["predicted_answer"] = scored_df["predicted_answer"].astype(str)
    scored_df["answer"] = scored_df["answer"].astype(str)

    scored_dataset = Dataset.from_pandas(scored_df[:])
    scored_dataset = llm_judge(
        scored_dataset,
    )

    scored_dataset.to_pandas().to_excel(fpath, index=False)

    print(fpath)

