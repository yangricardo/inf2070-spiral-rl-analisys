# Copyright 2025 SPIRAL Team. All Rights Reserved.
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

from typing import Optional


def apply_qwen3_template(observation: str, system_prompt: Optional[str] = None) -> str:
    del system_prompt
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(
    question: str, system_prompt: Optional[str] = None
) -> str:
    del system_prompt
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


TEMPLATE_FACTORY = {
    "qwen3": apply_qwen3_template,
    "qwen3_general": apply_qwen3_general_template,
}
