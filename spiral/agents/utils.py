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

import re


def kuhn_poker_parse_available_actions(observation: str):
    last_line = observation.strip().split("\n")[-1]
    available_actions = re.findall(r"\[(.*?)\]", last_line)

    # Add brackets
    available_actions = [f"[{action}]" for action in available_actions]
    # Remove [GAME]
    available_actions = [action for action in available_actions if action != "[GAME]"]
    return available_actions


def tic_tac_toe_parse_available_moves(observation: str):
    # Find the section after "Available Moves:" and before "Next Action:"
    moves_section_pattern = r"Available Moves:(.*?)Next Action:"
    moves_section = (
        re.search(moves_section_pattern, observation, re.DOTALL).group(1).strip()
    )

    # Now extract the moves from this section
    pattern = r"'\[(\d+)\]'"
    available_moves = re.findall(pattern, moves_section)

    available_moves = [f"[{move}]" for move in available_moves]

    return available_moves


_VALID_ACTION_PARSER = {
    "TicTacToe-v0": tic_tac_toe_parse_available_moves,
    "KuhnPoker-v1": kuhn_poker_parse_available_actions,
}


def get_valid_action_parser(env_id: str):
    try:
        return _VALID_ACTION_PARSER[env_id]
    except KeyError:
        raise NotImplementedError(f"valid action parser not implemented for {env_id}")
