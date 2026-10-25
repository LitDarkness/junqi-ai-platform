from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pygame
import torch

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

from JunqiEnv.junqi import PieceType, PlayerId

from junqi_rl.bots import (
    AlphaBetaAnalysis,
    BaseBot,
    build_bot,
    configure_heuristic_lookahead,
    score_heuristic_action_v2,
)
from junqi_rl.env.env_wrapper import JunqiPPOEnv
from junqi_rl.recording import GameRecord
from junqi_rl.rule_profile import default_rule_profile


BOARD_ROWS = 12
BOARD_COLS = 5
BOARD_SIZE = 60
SCREEN_W = 1320
SCREEN_H = 900
TOP_H = 52
BOTTOM_H = 164
LEFT_W = 220
RIGHT_W = 500
BOARD_W = SCREEN_W - LEFT_W - RIGHT_W
BOARD_H = SCREEN_H - TOP_H - BOTTOM_H
CELL_W = BOARD_W // BOARD_COLS
CELL_H = BOARD_H // BOARD_ROWS
BOARD_X = LEFT_W
BOARD_Y = TOP_H
TIMELINE_Y = BOARD_Y + BOARD_H

COLORS = {
    "background": (19, 24, 31),
    "top": (25, 32, 42),
    "panel": (30, 38, 49),
    "panel_alt": (35, 44, 57),
    "board": (207, 190, 157),
    "road": (125, 105, 78),
    "rail": (91, 72, 52),
    "station": (226, 211, 180),
    "camp": (133, 177, 169),
    "hq": (184, 152, 99),
    "red": (185, 58, 58),
    "red_dark": (114, 34, 36),
    "black": (55, 63, 75),
    "black_dark": (24, 29, 37),
    "text": (238, 241, 245),
    "muted": (160, 171, 185),
    "accent": (74, 178, 255),
    "green": (79, 205, 132),
    "yellow": (247, 198, 73),
    "danger": (245, 99, 99),
    "button": (47, 60, 77),
    "button_hover": (61, 79, 101),
}

# Define one home half and derive the opposite half by a 180-degree rotation.
# The old black-half constants were shifted down by one row (41/43/47/51/53),
# so the renderer disagreed with the C++ rules kernel and looked asymmetric.
RED_CAMPS = {11, 13, 17, 21, 23}
CAMPS = RED_CAMPS | {BOARD_SIZE - 1 - position for position in RED_CAMPS}
HQS = {1, 3, 56, 58}
RAIL_ROWS = {1, 5, 6, 10}
RAIL_COLS = {0, 4}
MOUNTAIN_GAPS = {frozenset((26, 31)), frozenset((28, 33))}
LAYOUT_DIRECTORY = Path(__file__).resolve().parent / "layouts"
EVOLVED_LAYOUT_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "junqi_layout_evolver" / "exports"
)

DEPLOYMENT_INVENTORY = Counter({
    90: 1,
    80: 1,
    70: 2,
    60: 2,
    50: 2,
    40: 2,
    30: 3,
    20: 3,
    10: 3,
    2: 2,
    1: 3,
    3: 1,
})

PIECE_NAMES = {
    PieceType.FLAG: "军旗",
    PieceType.MINE: "地雷",
    PieceType.BOMB: "炸弹",
    PieceType.SAPPER: "工兵",
    PieceType.LIEUTENANT: "排长",
    PieceType.CAPTAIN: "连长",
    PieceType.MAJOR: "营长",
    PieceType.COLONEL: "团长",
    PieceType.BRIGADIER: "旅长",
    PieceType.GENERAL: "师长",
    PieceType.ARMY_CMDR: "军长",
    PieceType.FIELD_MARSHAL: "司令",
}
PLAYER_NAMES = {PlayerId.RED: "红方", PlayerId.BLACK: "黑方", PlayerId.NONE: "无"}


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in (
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "DengXian",
        "SimHei",
        "Noto Sans CJK SC",
    ):
        font = pygame.font.SysFont(name, size, bold=bold)
        if font:
            return font
    return pygame.font.Font(None, size)


@dataclass
class Button:
    action: str
    label: str
    rect: pygame.Rect
    shortcut: str = ""


def _think_on_snapshot(
    bot: BaseBot,
    board: list[int],
    player: int,
    max_steps: int,
    initial_board: list[int] | None = None,
    moves: Sequence[dict] = (),
) -> int | None:
    wrapper = JunqiPPOEnv(max_steps=max_steps)
    if initial_board is not None:
        wrapper.env.load_board_state(initial_board, PlayerId.RED)
        for move in moves:
            wrapper.env.step(int(move["from"]), int(move["to"]))
        if list(wrapper.env.get_perfect_state()) != list(board):
            raise RuntimeError("snapshot history does not reconstruct the current board")
        if int(wrapper.env.get_current_player()) != int(player):
            raise RuntimeError("snapshot history does not reconstruct the current player")
        wrapper._az_initial_board = list(initial_board)
        wrapper._az_raw_moves = [
            (int(move["from"]), int(move["to"])) for move in moves
        ]
    else:
        wrapper.env.load_board_state(board, player)
    wrapper._current_player = player
    wrapper._prev_material_delta = wrapper._material_delta(player)
    return bot.select_action(wrapper, player)


def _deployment_errors(board: Sequence[int]) -> list[str]:
    if len(board) != BOARD_SIZE:
        return [f"棋盘长度必须为 {BOARD_SIZE}"]
    values = [int(value) for value in board]
    errors: list[str] = []
    if any(values[index] != 0 for index in CAMPS):
        errors.append("初始布局的十个行营必须为空")
    if any(values[index] < 0 for index in range(30)):
        errors.append("黑方棋子不能出现在红方布阵区")
    if any(values[index] > 0 for index in range(30, 60)):
        errors.append("红方棋子不能出现在黑方布阵区")

    red_counts = Counter(value for value in values if value > 0)
    black_counts = Counter(-value for value in values if value < 0)
    if red_counts != DEPLOYMENT_INVENTORY:
        errors.append("红方棋子数量或种类不正确")
    if black_counts != DEPLOYMENT_INVENTORY:
        errors.append("黑方棋子数量或种类不正确")

    red_flags = [index for index, value in enumerate(values) if value == int(PieceType.FLAG)]
    black_flags = [index for index, value in enumerate(values) if value == -int(PieceType.FLAG)]
    if len(red_flags) != 1 or red_flags[0] not in {1, 3}:
        errors.append("红方军旗必须位于 01 或 03 大本营")
    if len(black_flags) != 1 or black_flags[0] not in {56, 58}:
        errors.append("黑方军旗必须位于 56 或 58 大本营")

    for index, value in enumerate(values):
        row = index // BOARD_COLS
        if value == int(PieceType.MINE) and row not in {0, 1}:
            errors.append(f"红方地雷只能位于后两排：{index}")
        if value == -int(PieceType.MINE) and row not in {10, 11}:
            errors.append(f"黑方地雷只能位于后两排：{index}")
        if value == int(PieceType.BOMB) and row == 5:
            errors.append(f"红方炸弹不能位于最前排：{index}")
        if value == -int(PieceType.BOMB) and row == 6:
            errors.append(f"黑方炸弹不能位于最前排：{index}")
    return errors


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class JunqiWorkbench:
    def __init__(
        self,
        red_bot: BaseBot,
        black_bot: BaseBot,
        max_steps: int = 600,
        delay_ms: int = 250,
        engine_options: Sequence[str] = (),
        engine_resolver: Callable[[str], BaseBot] | None = None,
        red_engine_id: str | None = None,
        black_engine_id: str | None = None,
    ):
        pygame.init()
        pygame.display.set_caption("中国军棋明棋 AI 工作台")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
        self.canvas = pygame.Surface((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_title = load_font(25, True)
        self.font = load_font(19, True)
        self.font_small = load_font(16)
        self.font_tiny = load_font(14)

        self.env_wrapper = JunqiPPOEnv(max_steps=max_steps)
        self.env = self.env_wrapper.env
        self.red_bot = red_bot
        self.black_bot = black_bot
        self.engine_options = tuple(engine_options)
        self.engine_resolver = engine_resolver
        self.red_engine_id = red_engine_id or red_bot.name
        self.black_engine_id = black_engine_id or black_bot.name
        self.max_steps = max_steps
        self.delay_ms = delay_ms
        self.profile = default_rule_profile()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="junqi-ai")
        self.ai_future: concurrent.futures.Future | None = None
        self.ai_request: tuple[int, int, int] | None = None
        self.ai_started = 0.0
        self.next_ai_after = 0.0
        self.generation = 0

        self.auto_play = False
        self.step_once = False
        # Logical row 0 is RED's home rank. Put a human RED player at the
        # bottom by default; F still rotates the whole board by 180 degrees.
        self.view_flipped = bool(getattr(red_bot, "is_human", False))
        self.heatmap_enabled = False
        self.layout_edit_mode = False
        self.selected_pos: int | None = None
        self.drag_start: int | None = None
        self.hover_pos: int | None = None
        self.legal_destinations: list[int] = []
        self.heat_scores: dict[int, float] = {}
        self.heat_action_scores: dict[int, float] = {}
        self.heat_best_actions: dict[int, int] = {}
        self.heat_ranks: dict[int, int] = {}
        self.heat_source = ""
        self.last_move: tuple[int, int] | None = None
        self.last_result = "未开始"
        self.last_engine_ms = 0.0
        self.logs: list[str] = []
        self.moves: list[dict] = []
        self.snapshots: list[tuple[list[int], int]] = []
        self.timeline_cursor = 0
        self.initial_state: list[int] = []
        self.running = True
        self.buttons = self._make_buttons()
        self.reset()

    def _make_buttons(self) -> list[Button]:
        specs = [
            ("new", "新局", "R"),
            ("auto", "自动 / 暂停", "Space"),
            ("step", "单步", "→"),
            ("undo", "撤销", "Ctrl+Z"),
            ("redo", "重做", "Ctrl+Y"),
            ("flip", "翻转视角", "F"),
            ("heat", "策略热力图", "H"),
            ("engine_red", "切换红方引擎", "["),
            ("engine_black", "切换黑方引擎", "]"),
            ("layout_edit", "布局编辑", "L"),
            ("layout_save", "保存布局", "Ctrl+Shift+S"),
            ("layout_load", "载入布局", "Ctrl+Shift+O"),
            ("save", "保存记录", "Ctrl+S"),
            ("load", "载入最近记录", "Ctrl+O"),
            ("quit", "退出", "Esc"),
        ]
        buttons: list[Button] = []
        y = 112
        for action, label, shortcut in specs:
            buttons.append(Button(action, label, pygame.Rect(18, y, LEFT_W - 36, 30), shortcut))
            y += 34
        return buttons

    def _sync_wrapper(self) -> None:
        player = self.env.get_current_player()
        self.env_wrapper._current_player = player
        self.env_wrapper._prev_material_delta = self.env_wrapper._material_delta(player)

    def _invalidate_ai(self) -> None:
        self.generation += 1
        if self.ai_future and not self.ai_future.done():
            self.ai_future.cancel()
        self.ai_future = None
        self.ai_request = None

    def reset(self) -> None:
        self._invalidate_ai()
        self.layout_edit_mode = False
        self.env.reset_random()
        self._sync_wrapper()
        self.initial_state = list(self.env.get_perfect_state())
        self.moves = []
        self.snapshots = [(self.initial_state.copy(), self.env.get_current_player())]
        self.timeline_cursor = 0
        self.selected_pos = None
        self.legal_destinations = []
        self.heat_scores = {}
        self.heat_action_scores = {}
        self.heat_best_actions = {}
        self.heat_ranks = {}
        self.heat_source = ""
        self.last_move = None
        self.last_result = "新对局已开始"
        self.logs = [
            f"红方={self.red_bot.name}，黑方={self.black_bot.name}",
            f"规则={self.profile.profile_id}  {self.profile.sha256[:12]}…",
        ]
        self.next_ai_after = 0.0

    def current_bot(self) -> BaseBot:
        return self.red_bot if self.env.get_current_player() == PlayerId.RED else self.black_bot

    def _analysis_bot(self) -> BaseBot:
        current = self.current_bot()
        if not current.is_human:
            return current
        return self.black_bot if self.env.get_current_player() == PlayerId.RED else self.red_bot

    def _cycle_engine(self, player: int) -> None:
        if not self.engine_options or self.engine_resolver is None:
            self.log("当前入口没有提供可切换的引擎目录")
            return
        current_id = self.red_engine_id if player == PlayerId.RED else self.black_engine_id
        try:
            index = self.engine_options.index(current_id)
        except ValueError:
            index = -1
        next_id = self.engine_options[(index + 1) % len(self.engine_options)]
        try:
            bot = self.engine_resolver(next_id)
        except Exception as exc:
            self.log(f"切换引擎失败 {next_id}: {exc}")
            return
        self._invalidate_ai()
        if player == PlayerId.RED:
            self.red_bot = bot
            self.red_engine_id = next_id
        else:
            self.black_bot = bot
            self.black_engine_id = next_id
        self.selected_pos = None
        self._update_legal_moves()
        side = "红方" if player == PlayerId.RED else "黑方"
        self.log(f"{side}引擎已切换为 {next_id}: {bot.name}")

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-80:]

    def _piece_name(self, value: int) -> str:
        return "空地" if value == 0 else PIECE_NAMES.get(abs(value), str(abs(value)))

    def _cell_name(self, index: int) -> str:
        row, col = divmod(index, BOARD_COLS)
        return f"{index:02d}({row + 1},{col + 1})"

    def _logical_to_display(self, index: int) -> tuple[int, int]:
        row, col = divmod(index, BOARD_COLS)
        if self.view_flipped:
            row, col = BOARD_ROWS - 1 - row, BOARD_COLS - 1 - col
        return row, col

    def _display_to_logical(self, row: int, col: int) -> int:
        if self.view_flipped:
            row, col = BOARD_ROWS - 1 - row, BOARD_COLS - 1 - col
        return row * BOARD_COLS + col

    def _cell_rect(self, index: int) -> pygame.Rect:
        row, col = self._logical_to_display(index)
        return pygame.Rect(BOARD_X + col * CELL_W, BOARD_Y + row * CELL_H, CELL_W, CELL_H)

    def _cell_center(self, index: int) -> tuple[int, int]:
        return self._cell_rect(index).center

    def _mouse_cell(self, mouse: tuple[int, int]) -> int | None:
        x, y = mouse
        if not (BOARD_X <= x < BOARD_X + BOARD_W and BOARD_Y <= y < BOARD_Y + BOARD_H):
            return None
        col = (x - BOARD_X) // CELL_W
        row = (y - BOARD_Y) // CELL_H
        return self._display_to_logical(row, col)

    def _update_legal_moves(self) -> None:
        self.legal_destinations = []
        self.heat_scores = {}
        self.heat_action_scores = {}
        self.heat_best_actions = {}
        self.heat_ranks = {}
        self.heat_source = ""
        if self.layout_edit_mode:
            return
        if self.env.is_game_over():
            return
        current = self.env.get_current_player()
        all_actions = list(self.env.get_legal_actions(current))
        if self.selected_pos is not None:
            actions = [
                action for action in all_actions
                if action // BOARD_SIZE == self.selected_pos
            ]
        elif self.heatmap_enabled:
            actions = all_actions
        else:
            return
        self.legal_destinations = sorted({action % BOARD_SIZE for action in actions})
        if self.heatmap_enabled and actions:
            board = list(self.env.get_perfect_state())
            opponent = PlayerId.BLACK if current == PlayerId.RED else PlayerId.RED
            my_material = self.env.get_material_score(current)
            opp_material = self.env.get_material_score(opponent)
            analysis_bot = self._analysis_bot()
            scorer = getattr(analysis_bot, "score_actions", None)
            if callable(scorer):
                try:
                    scored = scorer(board, actions, int(current))
                    self.heat_action_scores = {
                        int(action): float(score)
                        for action, score in scored.items()
                        if int(action) in actions
                    }
                    self.heat_source = analysis_bot.name
                except Exception as exc:
                    self.log(f"引擎热力分析失败，使用通用评估: {exc}")
            if not self.heat_action_scores:
                self.heat_action_scores = {
                    int(action): float(score_heuristic_action_v2(
                        action, board, current, my_material, opp_material
                    ))
                    for action in actions
                }
                self.heat_source = "通用 heuristic_v2（当前引擎未暴露分数）"
            for action, score in self.heat_action_scores.items():
                target = action % BOARD_SIZE
                if target not in self.heat_scores or score > self.heat_scores[target]:
                    self.heat_scores[target] = score
                    self.heat_best_actions[target] = action
            ranked_targets = sorted(
                self.heat_scores, key=self.heat_scores.get, reverse=True
            )
            self.heat_ranks = {
                target: rank + 1 for rank, target in enumerate(ranked_targets)
            }

    def _select_or_move(self, index: int) -> None:
        if self.layout_edit_mode:
            self._select_or_swap_layout(index)
            return
        if self.env.is_game_over() or not self.current_bot().is_human:
            return
        board = self.env.get_perfect_state()
        value = board[index]
        owner = PlayerId.RED if value > 0 else PlayerId.BLACK if value < 0 else PlayerId.NONE
        current = self.env.get_current_player()
        if self.selected_pos is None:
            if owner == current:
                self.selected_pos = index
                self._update_legal_moves()
            return
        if index == self.selected_pos:
            self.selected_pos = None
            self._update_legal_moves()
            return
        if owner == current:
            self.selected_pos = index
            self._update_legal_moves()
            return

        result = self.env.check_move(self.selected_pos, index)
        if result.legal:
            self._execute_policy_action(self.selected_pos * BOARD_SIZE + index)
            self.selected_pos = None
            self._update_legal_moves()
        else:
            self.log(f"非法着 {self._cell_name(self.selected_pos)}→{self._cell_name(index)}：{result.code}")

    def _select_or_swap_layout(self, index: int) -> None:
        if self.moves or self.timeline_cursor != 0:
            self.log("布局编辑只允许在开局前进行")
            return
        board = list(self.env.get_perfect_state())
        value = board[index]
        if value == 0:
            self.log("请选择两个已有棋子进行交换")
            return
        if self.selected_pos is None:
            self.selected_pos = index
            self.log(f"已选择 {self._cell_name(index)} {self._piece_name(value)}")
            return
        if index == self.selected_pos:
            self.selected_pos = None
            return
        selected_value = board[self.selected_pos]
        if (selected_value > 0) != (value > 0):
            self.log("布局编辑只能交换同一方的两个棋子")
            return
        source = self.selected_pos
        board[source], board[index] = board[index], board[source]
        errors = _deployment_errors(board)
        if errors:
            self.log(f"布局交换被拒绝：{errors[0]}")
            return
        self._invalidate_ai()
        self.env.load_board_state(board, PlayerId.RED)
        self._sync_wrapper()
        self.initial_state = board.copy()
        self.moves = []
        self.snapshots = [(board.copy(), PlayerId.RED)]
        self.timeline_cursor = 0
        self.last_move = None
        self.selected_pos = None
        self.last_result = (
            f"布局交换 {self._cell_name(source)} ↔ {self._cell_name(index)}"
        )
        self.log(self.last_result)
        self._update_legal_moves()

    def toggle_layout_edit(self) -> None:
        if self.moves or self.timeline_cursor != 0:
            self.log("请先开始新局，再进入布局编辑")
            return
        self._invalidate_ai()
        self.auto_play = False
        self.step_once = False
        self.layout_edit_mode = not self.layout_edit_mode
        self.selected_pos = None
        self.heatmap_enabled = False
        self._update_legal_moves()
        self.log(
            "布局编辑已开启：依次点击两个同色棋子进行交换"
            if self.layout_edit_mode
            else "布局编辑已关闭，可以开始对局"
        )

    def save_layout(self) -> Path | None:
        board = list(self.env.get_perfect_state())
        errors = _deployment_errors(board)
        if errors:
            self.log(f"布局保存失败：{errors[0]}")
            return None
        path = LAYOUT_DIRECTORY / f"gui_layout_{time.strftime('%Y%m%d_%H%M%S')}.json"
        red_layout = board[:30]
        black_layout = [-board[59 - index] for index in range(30)]
        _atomic_json(path, {
            "schema": "junqi-layout-v1",
            "name": path.stem,
            "board": board,
            "red_layout": red_layout,
            "black_layout": black_layout,
            "metadata": {
                "source": "GUI",
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        })
        self.log(f"布局已保存：{path}")
        return path

    def load_layout(self, path: Path | None = None) -> Path | None:
        if path is None:
            candidates: list[Path] = []
            for directory in (LAYOUT_DIRECTORY, EVOLVED_LAYOUT_DIRECTORY):
                if directory.is_dir():
                    candidates.extend(directory.glob("*.json"))
            if not candidates:
                self.log("没有可载入的布局文件")
                return None
            path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("schema") != "junqi-layout-v1":
                raise ValueError("不支持的布局文件格式")
            board = [int(value) for value in payload["board"]]
            errors = _deployment_errors(board)
            if errors:
                raise ValueError(errors[0])
            self._invalidate_ai()
            self.auto_play = False
            self.step_once = False
            self.layout_edit_mode = False
            self.env.load_board_state(board, PlayerId.RED)
            self._sync_wrapper()
            self.initial_state = board.copy()
            self.moves = []
            self.snapshots = [(board.copy(), PlayerId.RED)]
            self.timeline_cursor = 0
            self.selected_pos = None
            self.last_move = None
            self.last_result = f"已载入布局 {Path(path).name}"
            self._update_legal_moves()
            self.log(f"布局已载入并通过规则校验：{Path(path)}")
            return Path(path)
        except Exception as exc:
            self.log(f"布局载入失败：{exc}")
            return None

    def _truncate_branch(self) -> None:
        if self.timeline_cursor < len(self.moves):
            self.moves = self.moves[: self.timeline_cursor]
            self.snapshots = self.snapshots[: self.timeline_cursor + 1]

    def _execute_policy_action(self, policy_action: int) -> None:
        if self.env.is_game_over():
            return
        self._truncate_branch()
        player = self.env.get_current_player()
        env_action = self.env_wrapper.action_to_env(policy_action, player)
        source, target = divmod(env_action, BOARD_SIZE)
        legality = self.env.check_move(source, target)
        if not legality.legal:
            self.log(f"引擎非法着被拒绝：{legality.code}")
            return

        before = list(self.env.get_perfect_state())
        before_key = int(self.env.get_state_key())
        source_name = self._piece_name(before[source])
        target_name = self._piece_name(before[target])
        _, reward, terminated, truncated, _ = self.env_wrapper.step(policy_action)
        self._sync_wrapper()
        after_key = int(self.env.get_state_key())
        move = {
            "ply": len(self.moves),
            "from": source,
            "to": target,
            "player": int(player),
            "source_piece": source_name,
            "target_piece": target_name,
            "reward": float(reward),
            "key_before": str(before_key),
            "key_after": str(after_key),
        }
        self.moves.append(move)
        self.snapshots.append((list(self.env.get_perfect_state()), self.env.get_current_player()))
        self.timeline_cursor = len(self.moves)
        self.last_move = (source, target)
        actor = PLAYER_NAMES.get(player, "未知方")
        self.last_result = (
            f"{actor} {self._cell_name(source)} {source_name} → "
            f"{self._cell_name(target)} {target_name}"
        )
        self.log(f"{len(self.moves):03d} {self.last_result}")
        if terminated or truncated:
            self.log(self._terminal_text())
        self.next_ai_after = time.perf_counter() + self.delay_ms / 1000.0
        self.selected_pos = None
        self._update_legal_moves()

    def _terminal_text(self) -> str:
        winner = self.env.get_winner()
        reason = self.env.get_terminal_reason()
        if winner == PlayerId.NONE:
            return f"终局：平局（{reason}）"
        return f"终局：{PLAYER_NAMES[winner]}获胜（{reason}）"

    def _load_snapshot(self, cursor: int) -> None:
        cursor = max(0, min(cursor, len(self.snapshots) - 1))
        self._invalidate_ai()
        board, player = self.snapshots[cursor]
        self.env.load_board_state(self.initial_state, PlayerId.RED)
        for move in self.moves[:cursor]:
            self.env.step(int(move["from"]), int(move["to"]))
        if list(self.env.get_perfect_state()) != list(board):
            raise RuntimeError("timeline replay does not match its stored snapshot")
        if int(self.env.get_current_player()) != int(player):
            raise RuntimeError("timeline replay has the wrong current player")
        self._sync_wrapper()
        self.layout_edit_mode = False
        self.timeline_cursor = cursor
        self.last_move = None if cursor == 0 else (
            self.moves[cursor - 1]["from"],
            self.moves[cursor - 1]["to"],
        )
        self.selected_pos = None
        self._update_legal_moves()
        self.log(f"时间线跳转到 ply {cursor}；继续走子将建立新分支")

    def undo(self) -> None:
        if self.timeline_cursor > 0:
            self._load_snapshot(self.timeline_cursor - 1)

    def redo(self) -> None:
        if self.timeline_cursor < len(self.snapshots) - 1:
            self._load_snapshot(self.timeline_cursor + 1)

    def save_record(self) -> Path:
        record = GameRecord(
            self.profile,
            self.initial_state,
            [
                {"engine": self.red_bot.name, "side": "red"},
                {"engine": self.black_bot.name, "side": "black"},
            ],
            seed=0,
        )
        for move in self.moves:
            record.append(
                ply=move["ply"],
                from_pos=move["from"],
                to_pos=move["to"],
                key_before=move["key_before"],
                key_after=move["key_after"],
                event={"type": "MOVE", "reward": move["reward"]},
            )
        if self.env.is_game_over():
            record.result = {
                "winner": int(self.env.get_winner()),
                "reason": self.env.get_terminal_reason(),
            }
        path = Path("records") / f"gui_{time.strftime('%Y%m%d_%H%M%S')}.json"
        record.save_atomic(path)
        self.log(f"记录已保存：{path}")
        return path

    def load_latest_record(self) -> None:
        candidates = sorted(Path("records").glob("gui_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            self.log("没有可载入的 GUI 记录")
            return
        path = candidates[-1]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("rule_profile_sha256") != self.profile.sha256:
                raise ValueError("RuleProfile SHA-256 不匹配")
            self._invalidate_ai()
            self.layout_edit_mode = False
            initial = [int(value) for value in payload["initial_state"]]
            self.env.load_board_state(initial, PlayerId.RED)
            self._sync_wrapper()
            self.initial_state = initial
            self.moves = []
            self.snapshots = [(initial.copy(), PlayerId.RED)]
            for raw in payload.get("moves", []):
                before_key = str(self.env.get_state_key())
                if before_key != str(raw.get("key_before")):
                    raise ValueError(f"ply {raw.get('ply')} key_before 不匹配")
                source, target = int(raw["from"]), int(raw["to"])
                player = self.env.get_current_player()
                self.env.step(source, target)
                after_key = str(self.env.get_state_key())
                if after_key != str(raw.get("key_after")):
                    raise ValueError(f"ply {raw.get('ply')} key_after 不匹配")
                self.moves.append({
                    "ply": int(raw["ply"]),
                    "from": source,
                    "to": target,
                    "player": int(player),
                    "source_piece": "",
                    "target_piece": "",
                    "reward": float((raw.get("event") or {}).get("reward", 0.0)),
                    "key_before": before_key,
                    "key_after": after_key,
                })
                self.snapshots.append((list(self.env.get_perfect_state()), self.env.get_current_player()))
            self._sync_wrapper()
            self.timeline_cursor = len(self.moves)
            self.last_move = None if not self.moves else (self.moves[-1]["from"], self.moves[-1]["to"])
            self.log(f"记录载入并校验通过：{path}")
        except Exception as exc:
            self.log(f"载入失败：{exc}")

    def _start_ai(self) -> None:
        if self.layout_edit_mode or self.ai_future is not None or self.env.is_game_over():
            return
        bot = self.current_bot()
        if bot.is_human:
            return
        if not (self.auto_play or self.step_once):
            return
        if not self.step_once and time.perf_counter() < self.next_ai_after:
            return
        player = self.env.get_current_player()
        key = int(self.env.get_state_key())
        self.ai_request = (self.generation, key, int(player))
        self.ai_started = time.perf_counter()
        self.ai_future = self.executor.submit(
            _think_on_snapshot,
            bot,
            list(self.env.get_perfect_state()),
            player,
            self.max_steps,
            self.initial_state.copy(),
            [dict(move) for move in self.moves[: self.timeline_cursor]],
        )

    def _poll_ai(self) -> None:
        if self.ai_future is None or not self.ai_future.done():
            return
        future, request = self.ai_future, self.ai_request
        self.ai_future = None
        self.ai_request = None
        self.last_engine_ms = (time.perf_counter() - self.ai_started) * 1000.0
        try:
            action = future.result()
        except Exception as exc:
            self.auto_play = False
            self.log(f"引擎异常，已暂停：{exc}")
            return
        if request is None:
            return
        generation, key, player = request
        if (
            generation != self.generation
            or key != int(self.env.get_state_key())
            or player != int(self.env.get_current_player())
        ):
            self.log("已丢弃过期引擎响应")
            return
        if action is not None:
            self._execute_policy_action(int(action))
        self.step_once = False

    def _handle_action(self, action: str) -> None:
        if action == "new":
            self.reset()
        elif action == "auto":
            self.auto_play = not self.auto_play
        elif action == "step":
            self.step_once = True
        elif action == "undo":
            self.undo()
        elif action == "redo":
            self.redo()
        elif action == "flip":
            self.view_flipped = not self.view_flipped
        elif action == "heat":
            self.heatmap_enabled = not self.heatmap_enabled
            self._update_legal_moves()
            if self.heatmap_enabled:
                self.log("热力图已开启：无需选子显示全局，选子后聚焦该棋子的合法着")
        elif action == "engine_red":
            self._cycle_engine(PlayerId.RED)
        elif action == "engine_black":
            self._cycle_engine(PlayerId.BLACK)
        elif action == "layout_edit":
            self.toggle_layout_edit()
        elif action == "layout_save":
            self.save_layout()
        elif action == "layout_load":
            self.load_layout()
        elif action == "save":
            self.save_record()
        elif action == "load":
            self.load_latest_record()
        elif action == "quit":
            self.running = False

    def _draw_top(self) -> None:
        pygame.draw.rect(self.canvas, COLORS["top"], (0, 0, SCREEN_W, TOP_H))
        title = self.font_title.render("中国军棋明棋 AI 工作台", True, COLORS["text"])
        self.canvas.blit(title, (18, 11))
        status = (
            "布局编辑"
            if self.layout_edit_mode
            else "思考中"
            if self.ai_future
            else "自动运行"
            if self.auto_play
            else "已暂停"
        )
        status_color = (
            COLORS["accent"]
            if self.layout_edit_mode
            else COLORS["yellow"]
            if self.ai_future
            else COLORS["green"]
            if self.auto_play
            else COLORS["muted"]
        )
        status_surface = self.font.render(status, True, status_color)
        self.canvas.blit(status_surface, (SCREEN_W - status_surface.get_width() - 22, 15))

    def _draw_left(self, mouse: tuple[int, int]) -> None:
        pygame.draw.rect(self.canvas, COLORS["panel"], (0, TOP_H, LEFT_W, BOARD_H))
        self.canvas.blit(self.font.render("对局控制", True, COLORS["accent"]), (18, 76))
        for button in self.buttons:
            hover = button.rect.collidepoint(mouse)
            color = COLORS["button_hover"] if hover else COLORS["button"]
            if button.action == "auto" and self.auto_play:
                color = (46, 111, 85)
            if button.action == "heat" and self.heatmap_enabled:
                color = (78, 88, 139)
            if button.action == "layout_edit" and self.layout_edit_mode:
                color = (126, 91, 48)
            pygame.draw.rect(self.canvas, color, button.rect, border_radius=8)
            self.canvas.blit(self.font_small.render(button.label, True, COLORS["text"]), (button.rect.x + 12, button.rect.y + 7))
            shortcut = self.font_tiny.render(button.shortcut, True, COLORS["muted"])
            self.canvas.blit(shortcut, (button.rect.right - shortcut.get_width() - 9, button.rect.y + 9))

        y = 630
        for line in (
            f"红方  {self.red_bot.name}",
            f"黑方  {self.black_bot.name}",
            f"引擎耗时  {self.last_engine_ms:.1f} ms",
        ):
            self.canvas.blit(self.font_small.render(line, True, COLORS["muted"]), (18, y))
            y += 24

    def _draw_topology(self) -> None:
        pygame.draw.rect(self.canvas, COLORS["board"], (BOARD_X, BOARD_Y, BOARD_W, BOARD_H))
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                source = row * BOARD_COLS + col
                for dr, dc in ((1, 0), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if nr >= BOARD_ROWS or nc >= BOARD_COLS:
                        continue
                    target = nr * BOARD_COLS + nc
                    if frozenset((source, target)) in MOUNTAIN_GAPS:
                        continue
                    pygame.draw.line(self.canvas, COLORS["road"], self._cell_center(source), self._cell_center(target), 2)
        for camp in CAMPS:
            row, col = divmod(camp, BOARD_COLS)
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                        pygame.draw.line(self.canvas, COLORS["road"], self._cell_center(camp), self._cell_center(nr * 5 + nc), 2)

        for row in RAIL_ROWS:
            pygame.draw.line(self.canvas, COLORS["rail"], self._cell_center(row * 5), self._cell_center(row * 5 + 4), 6)
        for col in RAIL_COLS:
            pygame.draw.line(self.canvas, COLORS["rail"], self._cell_center(5 + col), self._cell_center(50 + col), 6)
        for col in (0, 2, 4):
            pygame.draw.line(self.canvas, COLORS["rail"], self._cell_center(25 + col), self._cell_center(30 + col), 6)

    def _heat_color(self, target: int) -> tuple[int, int, int]:
        if not self.heat_scores:
            return COLORS["green"]
        values = list(self.heat_scores.values())
        low, high = min(values), max(values)
        ratio = 0.5 if high <= low else (self.heat_scores[target] - low) / (high - low)
        return (int(240 - 160 * ratio), int(100 + 120 * ratio), 75)

    def _draw_board(self) -> None:
        self._draw_topology()
        board = self.env.get_perfect_state()
        for index in range(BOARD_SIZE):
            rect = self._cell_rect(index)
            center = rect.center
            if index in CAMPS:
                points = [(center[0], center[1] - 17), (center[0] + 24, center[1]), (center[0], center[1] + 17), (center[0] - 24, center[1])]
                pygame.draw.polygon(self.canvas, COLORS["camp"], points)
                pygame.draw.polygon(self.canvas, COLORS["road"], points, 2)
            elif index in HQS:
                station = pygame.Rect(0, 0, 50, 32)
                station.center = center
                pygame.draw.rect(self.canvas, COLORS["hq"], station, border_radius=7)
                pygame.draw.rect(self.canvas, COLORS["road"], station, 2, border_radius=7)
            else:
                pygame.draw.circle(self.canvas, COLORS["station"], center, 12)
                pygame.draw.circle(self.canvas, COLORS["road"], center, 12, 2)

            if index in self.legal_destinations:
                color = self._heat_color(index)
                if self.heatmap_enabled and index in self.heat_scores:
                    overlay = pygame.Surface((CELL_W, CELL_H), pygame.SRCALPHA)
                    pygame.draw.circle(
                        overlay, (*color, 105), (CELL_W // 2, CELL_H // 2), 25
                    )
                    self.canvas.blit(overlay, rect.topleft)
                pygame.draw.circle(self.canvas, color, center, 21, 5)

            value = board[index]
            if value:
                piece_rect = pygame.Rect(0, 0, 70, 38)
                piece_rect.center = center
                shadow = piece_rect.move(3, 4)
                pygame.draw.rect(self.canvas, (20, 20, 20, 80), shadow, border_radius=8)
                main = COLORS["red"] if value > 0 else COLORS["black"]
                border = COLORS["red_dark"] if value > 0 else COLORS["black_dark"]
                pygame.draw.rect(self.canvas, main, piece_rect, border_radius=8)
                pygame.draw.rect(self.canvas, border, piece_rect, 3, border_radius=8)
                label = self.font.render(self._piece_name(value), True, COLORS["text"])
                self.canvas.blit(label, label.get_rect(center=center))

            if self.heatmap_enabled and index in self.heat_ranks:
                rank = self.heat_ranks[index]
                if self.selected_pos is not None or rank <= 10:
                    badge_center = (rect.right - 16, rect.top + 15)
                    pygame.draw.circle(self.canvas, COLORS["top"], badge_center, 13)
                    rank_surface = self.font_tiny.render(str(rank), True, COLORS["yellow"])
                    self.canvas.blit(rank_surface, rank_surface.get_rect(center=badge_center))

            if self.last_move and index in self.last_move:
                pygame.draw.rect(self.canvas, COLORS["yellow"], rect.inflate(-8, -6), 3, border_radius=8)
            if index == self.selected_pos:
                pygame.draw.rect(self.canvas, COLORS["accent"], rect.inflate(-5, -4), 4, border_radius=8)
            if index == self.hover_pos:
                pygame.draw.rect(self.canvas, COLORS["text"], rect.inflate(-3, -2), 1, border_radius=8)

    def _analysis_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        player = self.env.get_current_player()
        legal_count = (
            0
            if self.layout_edit_mode or self.env.is_game_over()
            else len(self.env.get_legal_actions(player))
        )
        lines: list[tuple[str, tuple[int, int, int]]] = [
            ("规则与状态", COLORS["accent"]),
            (f"RuleProfile  {self.profile.profile_id}", COLORS["text"]),
            (f"规则哈希  {self.profile.sha256[:20]}…", COLORS["muted"]),
            (
                f"模式  {'布局编辑（点击两个同色棋子交换）' if self.layout_edit_mode else '正常对局'}",
                COLORS["yellow"] if self.layout_edit_mode else COLORS["text"],
            ),
            (f"行棋方  {PLAYER_NAMES.get(player, '未知')}", COLORS["text"]),
            (f"ply  {self.env.get_turn_count()} / {self.max_steps}", COLORS["text"]),
            (f"合法着  {legal_count}", COLORS["text"]),
            (f"状态键  {int(self.env.get_state_key()):016x}", COLORS["muted"]),
            (f"终局原因  {self.env.get_terminal_reason()}", COLORS["text"]),
            (f"子力  红 {self.env.get_material_score(PlayerId.RED):.1f}  黑 {self.env.get_material_score(PlayerId.BLACK):.1f}", COLORS["text"]),
            ("", COLORS["text"]),
            ("搜索分析", COLORS["accent"]),
        ]
        analysis = getattr(self.current_bot(), "last_analysis", None)
        if isinstance(analysis, AlphaBetaAnalysis):
            best = "—" if analysis.best_action is None else self._format_action(analysis.best_action)
            pv = " ".join(self._format_action(action) for action in analysis.pv[:5]) or "—"
            lines.extend([
                (f"best move  {best}", COLORS["yellow"]),
                (f"score  {analysis.score:.2f}（当前方静态分）", COLORS["text"]),
                (f"depth  {analysis.depth}  completed={analysis.completed}", COLORS["text"]),
                (f"nodes  {analysis.nodes} + q{analysis.qnodes}", COLORS["text"]),
                (f"NPS  {analysis.nps:,.0f}", COLORS["text"]),
                (f"TT hit  {analysis.tt_hits}  cutoff {analysis.cutoffs}", COLORS["text"]),
                (f"PV  {pv}", COLORS["muted"]),
            ])
        else:
            lines.extend([
                (f"当前引擎  {self.current_bot().name}", COLORS["text"]),
                ("该引擎未提供搜索树统计", COLORS["muted"]),
            ])

        if self.heatmap_enabled:
            lines.extend([
                ("", COLORS["text"]),
                ("策略热力图", COLORS["accent"]),
                (f"分析源  {self.heat_source or '等待合法局面'}", COLORS["text"]),
            ])
            for target, rank in sorted(self.heat_ranks.items(), key=lambda item: item[1])[:3]:
                action = self.heat_best_actions[target]
                lines.append((
                    f"#{rank}  {self._format_action(action)}  {self.heat_scores[target]:.3f}",
                    COLORS["yellow"] if rank == 1 else COLORS["muted"],
                ))

        if self.hover_pos is not None:
            value = self.env.get_perfect_state()[self.hover_pos]
            terrain = "行营" if self.hover_pos in CAMPS else "大本营" if self.hover_pos in HQS else "兵站"
            lines.extend([
                ("", COLORS["text"]),
                ("悬停检查", COLORS["accent"]),
                (f"位置  {self._cell_name(self.hover_pos)}  {terrain}", COLORS["text"]),
                (f"棋子  {self._piece_name(value)}", COLORS["text"]),
            ])
            if self.selected_pos is not None:
                if self.layout_edit_mode:
                    selected_value = self.env.get_perfect_state()[self.selected_pos]
                    same_side = value != 0 and (selected_value > 0) == (value > 0)
                    lines.append((
                        "布局交换  可交换" if same_side else "布局交换  请选择同色棋子",
                        COLORS["green"] if same_side else COLORS["danger"],
                    ))
                else:
                    result = self.env.check_move(self.selected_pos, self.hover_pos)
                    lines.append((f"合法性  {result.code}", COLORS["green"] if result.legal else COLORS["danger"]))
        return lines

    def _format_action(self, action: int) -> str:
        source, target = divmod(action, BOARD_SIZE)
        return f"{source:02d}→{target:02d}"

    def _draw_right(self) -> None:
        x = BOARD_X + BOARD_W
        pygame.draw.rect(self.canvas, COLORS["panel"], (x, TOP_H, RIGHT_W, BOARD_H))
        y = 70
        for text, color in self._analysis_lines():
            if not text:
                y += 8
                continue
            surface = self.font_small.render(text, True, color)
            self.canvas.blit(surface, (x + 18, y))
            y += 24

        log_y = 502
        pygame.draw.rect(self.canvas, COLORS["panel_alt"], (x + 12, log_y, RIGHT_W - 24, BOARD_Y + BOARD_H - log_y - 10), border_radius=9)
        self.canvas.blit(self.font.render("实时日志", True, COLORS["accent"]), (x + 24, log_y + 12))
        y = log_y + 43
        for line in self.logs[-7:]:
            surface = self.font_tiny.render(line[:64], True, COLORS["text"])
            self.canvas.blit(surface, (x + 24, y))
            y += 20

    def _draw_timeline(self, mouse: tuple[int, int]) -> None:
        pygame.draw.rect(self.canvas, COLORS["top"], (0, TIMELINE_Y, SCREEN_W, BOTTOM_H))
        self.canvas.blit(self.font.render("对局时间线", True, COLORS["accent"]), (18, TIMELINE_Y + 12))
        start = max(0, self.timeline_cursor - 11)
        shown = self.moves[start:start + 12]
        for offset, move in enumerate(shown):
            cursor = start + offset + 1
            rect = pygame.Rect(18 + offset * 106, TIMELINE_Y + 48, 98, 66)
            color = COLORS["button_hover"] if cursor == self.timeline_cursor else COLORS["button"]
            if rect.collidepoint(mouse):
                color = (75, 94, 119)
            pygame.draw.rect(self.canvas, color, rect, border_radius=7)
            player_color = COLORS["red"] if move["player"] == PlayerId.RED else COLORS["muted"]
            self.canvas.blit(self.font_tiny.render(f"ply {cursor}", True, player_color), (rect.x + 8, rect.y + 7))
            self.canvas.blit(self.font_small.render(f"{move['from']:02d}→{move['to']:02d}", True, COLORS["text"]), (rect.x + 8, rect.y + 29))
        summary = self.font_tiny.render(self.last_result[:88], True, COLORS["muted"])
        self.canvas.blit(summary, (18, TIMELINE_Y + 126))

    def draw(self) -> None:
        mouse_window = pygame.mouse.get_pos()
        window_size = self.screen.get_size()
        mouse = (
            int(mouse_window[0] * SCREEN_W / max(1, window_size[0])),
            int(mouse_window[1] * SCREEN_H / max(1, window_size[1])),
        )
        self.hover_pos = self._mouse_cell(mouse)
        self.canvas.fill(COLORS["background"])
        self._draw_top()
        self._draw_left(mouse)
        self._draw_board()
        self._draw_right()
        self._draw_timeline(mouse)
        if self.env.is_game_over():
            overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            overlay.fill((9, 12, 17, 165))
            self.canvas.blit(overlay, (BOARD_X, BOARD_Y))
            label = self.font_title.render(self._terminal_text(), True, COLORS["yellow"])
            self.canvas.blit(label, label.get_rect(center=(BOARD_X + BOARD_W // 2, BOARD_Y + BOARD_H // 2)))
        scaled = pygame.transform.smoothscale(self.canvas, self.screen.get_size())
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def _scaled_mouse(self, pos: tuple[int, int]) -> tuple[int, int]:
        size = self.screen.get_size()
        return (
            int(pos[0] * SCREEN_W / max(1, size[0])),
            int(pos[1] * SCREEN_H / max(1, size[1])),
        )

    def _timeline_click(self, pos: tuple[int, int]) -> bool:
        if pos[1] < TIMELINE_Y + 48 or pos[1] > TIMELINE_Y + 114:
            return False
        offset = (pos[0] - 18) // 106
        start = max(0, self.timeline_cursor - 11)
        cursor = start + offset + 1
        if 0 <= offset < 12 and 1 <= cursor <= len(self.moves):
            self._load_snapshot(cursor)
            return True
        return False

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            ctrl = bool(event.mod & pygame.KMOD_CTRL)
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self._handle_action("auto")
            elif event.key == pygame.K_RIGHT:
                self._handle_action("redo" if self.timeline_cursor < len(self.moves) else "step")
            elif event.key == pygame.K_LEFT:
                self._handle_action("undo")
            elif event.key == pygame.K_HOME:
                self._load_snapshot(0)
            elif event.key == pygame.K_END:
                self._load_snapshot(len(self.snapshots) - 1)
            elif event.key == pygame.K_r:
                self._handle_action("new")
            elif event.key == pygame.K_f:
                self._handle_action("flip")
            elif event.key == pygame.K_h:
                self._handle_action("heat")
            elif event.key == pygame.K_l:
                self._handle_action("layout_edit")
            elif event.key == pygame.K_LEFTBRACKET:
                self._handle_action("engine_red")
            elif event.key == pygame.K_RIGHTBRACKET:
                self._handle_action("engine_black")
            elif ctrl and event.key == pygame.K_z:
                self._handle_action("undo")
            elif ctrl and event.key == pygame.K_y:
                self._handle_action("redo")
            elif ctrl and bool(event.mod & pygame.KMOD_SHIFT) and event.key == pygame.K_s:
                self._handle_action("layout_save")
            elif ctrl and bool(event.mod & pygame.KMOD_SHIFT) and event.key == pygame.K_o:
                self._handle_action("layout_load")
            elif ctrl and event.key == pygame.K_s:
                self._handle_action("save")
            elif ctrl and event.key == pygame.K_o:
                self._handle_action("load")
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = self._scaled_mouse(event.pos)
            for button in self.buttons:
                if button.rect.collidepoint(pos):
                    self._handle_action(button.action)
                    return
            if self._timeline_click(pos):
                return
            self.drag_start = self._mouse_cell(pos)
            if self.drag_start is not None:
                self._select_or_move(self.drag_start)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pos = self._scaled_mouse(event.pos)
            target = self._mouse_cell(pos)
            if self.drag_start is not None and target is not None and target != self.drag_start:
                if self.selected_pos != self.drag_start:
                    self.selected_pos = self.drag_start
                    self._update_legal_moves()
                self._select_or_move(target)
            self.drag_start = None

    def run(self, max_frames: int | None = None, screenshot_path: Path | None = None) -> None:
        frames = 0
        try:
            while self.running:
                for event in pygame.event.get():
                    self._handle_event(event)
                self._poll_ai()
                self._start_ai()
                self.draw()
                self.clock.tick(60)
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
            if screenshot_path is not None:
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(self.canvas, screenshot_path)
        finally:
            self._invalidate_ai()
            self.executor.shutdown(wait=False, cancel_futures=True)
            pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="中国军棋明棋 AI 可视化工作台")
    parser.add_argument("--red", default="human", help="human | random | greedy | heuristic | heuristic_v1 | heuristic_v2 | heuristic_v3 | ab | PPO 路径")
    parser.add_argument("--black", default="heuristic", help="human | random | greedy | heuristic | heuristic_v1 | heuristic_v2 | heuristic_v3 | ab | PPO 路径")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--delay-ms", type=int, default=250)
    parser.add_argument("--heuristic-candidates", type=int, default=4)
    parser.add_argument("--heuristic-replies", type=int, default=3)
    parser.add_argument("--heuristic-counters", type=int, default=2)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--headless-smoke", action="store_true")
    args = parser.parse_args()

    if args.headless_smoke:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    configure_heuristic_lookahead(
        candidates=args.heuristic_candidates,
        replies=args.heuristic_replies,
        counters=args.heuristic_counters,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red_bot = build_bot(args.red, device)
    black_bot = build_bot(args.black, device)
    workbench = JunqiWorkbench(red_bot, black_bot, args.max_steps, args.delay_ms)
    if args.layout:
        loaded = workbench.load_layout(args.layout)
        if loaded is None:
            raise SystemExit(f"failed to load layout: {args.layout}")
    if args.headless_smoke:
        workbench.auto_play = True
        workbench.run(max_frames=45, screenshot_path=Path("reports/gui_smoke.png"))
        print("GUI_HEADLESS_SMOKE_PASS")
    else:
        workbench.run()


if __name__ == "__main__":
    main()
