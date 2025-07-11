"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from typing import Dict

import torch

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from transformers.utils import logging

from transformers.utils.import_utils import _is_package_available
# Use same as transformers.utils.import_utils


logger = logging.get_logger(__name__)
logging.set_verbosity(verbosity=logging.INFO)   # verbosity: int


def is_e2b_available() -> bool:
    return _is_package_available("e2b")

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()


from utils import is_messages

# #####################
# # Check if messages
# #####################
# # Adapted from: https://github.com/huggingface/trl/blob/v0.15.1/trl/data_utils.py#L24
# def is_messages(examples):
#     # examples = [
#     #     {"role": "user", "content": "What color is the sky?"},
#     #     {"role": "assitant", "content": "The sky is blue."}
#     # ]
#     # It must be a list of messages.
#     if isinstance(examples, list):
#         maybe_message = examples[0]
#         # Each message must a list of dictionaries with keys "role" and "content"
#         if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
#             return True
#     return False



def accuracy_reward(completions, solutions, answers, **kwargs):
    """
    Reward function that checks if the completion is the same as the ground truth.
    
    Note: `solutions` may be passed by kwargs (**reward_kwargs) and is a list of solution text.
        >> keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        >> reward_kwargs = {key: [example[key] for example in inputs] for key in keys}     # reward_kwargs contains 'solutions'
        >> output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    
    math_verify.parser.parse:
        Ref: 
            https://github.com/huggingface/Math-Verify/blob/0.5.2/src/math_verify/parser.py#L573
        
        Args: 
            pred (str): The prediction string to parse.
            extraction_config (Sequence[ExtractionTarget], optional): Configuration for what types of expressions 
                to extract and how to extract them. Defaults to [LatexExtractionConfig(), ExprExtractionConfig()].
            
            extraction_mode (Literal["first_match", "any_match"], optional): Strategy for extracting matches. Defaults to "any_match".
                - "first_match": Stop after finding the first match
                - "any_match": Try to extract all possible matches
    
    math_verify.grader.verify
        Verifies if the target expression matches the gold expression using multiple comparison strategies.
        Ref:
            https://github.com/huggingface/Math-Verify/blob/0.5.2/src/math_verify/grader.py#L602

        Args:
             gold: The reference/correct expression(s). Can be:
                - A single SymPy expression (Basic or MatrixBase)
                - A string
                - A list of any of the above
            target: The expression(s) to verify. Same types as gold.
            precision: Number of decimal places to consider for numeric comparisons. Defaults to 6.
    """
    
    if is_messages(completions[0]):     # message format
        completions = [example[0]["content"] for example in completions]
        # completions = [example["content"] for example in completions]
    
    # # Ground truth answers/solutions
    # # 'answers' are preferred over 'solutions'
    # if answers is not None:
    #     ground_truth = answers
    # elif solutions is not None:
    #     ground_truth = solutions
    # else:
    #     raise ValueError("No ground truth provided! Please check the data.")
    
    rewards = []
    for completion_to_parse, answer_to_parse, solution_to_parse in zip(completions, answers, solutions):
        
        # Parse gold/ground truth from both answer and solution text
        gold_parsed = []
        if answer_to_parse is not None:
            gold_parsed += parse(answer_to_parse)
        if solution_to_parse is not None:
            gold_parsed += parse(solution_to_parse)
        
        if len(gold_parsed) != 0:
            # Parse completion
            # Loose mode:
            # TODO: may extract from answer block (e.g., <answer>...</answer>) only
            answer_parsed = parse(completion_to_parse)
            
            # Strict mode:
            # We require the answer to be provided in correct latex (no malformed operators)
            # answer_parsed = parse(
            #     completion_to_parse,
            #     extraction_config=[
            #         LatexExtractionConfig(
            #             normalization_config=NormalizationConfig(
            #                 nits=False,
            #                 malformed_operators=False,
            #                 basic_latex=True,
            #                 equations=True,
            #                 boxed="all",
            #                 units=True,
            #             ),
            #             # Ensures that boxed is tried first
            #             boxed_match_priority=0,
            #             try_extract_without_anchor=False,
            #         )
            #     ],
            #     extraction_mode="first_match",
            # )

            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(gold=gold_parsed, target=answer_parsed))
            except Exception as e:
                logger.error(f"Verify failed: {e}, completion: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable,             
            # we reward 0 to skip this example
            reward = 0.0
            logger.error(f"Failed to parse gold solution: {answer_to_parse=}, {solution_to_parse=}.")
        rewards.append(reward)
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    
    # Completion text (may extract from conversional completions)
    if is_messages(completions[0]):
        # message format and only one completion each
        completions_text = [example[0]['content'] for example in completions]
        # completions_text = [example['content'] for example in completions]
    else:
        completions_text = completions
    
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    # Allow `\s`, `\n`, `\t` before `<think>` and after `</answer>`?
    pattern = r"^[\s]*<think>.+?</think>[\s]*<answer>.+?</answer>[\s]*$"
    
    # Multiline mode:  `$`` may match the end of each line instead of the end of the entire string
    # matches = [re.match(pattern, text, re.DOTALL | re.MULTILINE) for text in completions_text]
    
    # Match the end of the entire string instead of the end of each line
    matches = [re.match(pattern, text, re.DOTALL) for text in completions_text]
    
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`."""
    
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count
    
    # Completion text (may extract from conversional completions)
    if is_messages(completions[0]):
        # message format and only one completion each
        completions_text = [example[0]['content'] for example in completions]
        # completions_text = [example['content'] for example in completions]
    else:
        completions_text = completions
    
    return [count_tags(text) for text in completions_text]



def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]



# Note: parse and verify need to be checked in the following reward functions!

def len_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solutions, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solutions: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solutions):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()
            if output.strip() == case["output"].strip():
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]
    try:
        rewards = run_async_from_sync(scripts, verification_info["language"])

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the async function and get the result
        rewards = loop.run_until_complete(run_async(scripts, language))
    finally:
        loop.close()

    return rewards


async def run_async(scripts: list[str], language: str) -> list[float]:
    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards


async def run_script(sbx, script: str, language: str) -> float:
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0