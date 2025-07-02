# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Callbacks to use with the Trainer class and customize the training loop.
"""


from tqdm.auto import tqdm

from transformers.trainer_utils import has_length
from transformers.trainer_callback import TrainerCallback

class GRPOProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len
    
    def on_train_begin(self, args, state, control, **kwargs):
        desc=kwargs.get("desc", "Optimizing step")
        if state.is_world_process_zero:
            self.training_bar = tqdm(
                # total=state.max_steps,
                dynamic_ncols=True, 
                desc=desc
            )
        self.current_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                desc=kwargs.get("desc", "Evaluating")
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), 
                    leave=self.training_bar is None, dynamic_ncols=True,
                    desc=desc
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            self.training_bar.write(str(shallow_logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None

