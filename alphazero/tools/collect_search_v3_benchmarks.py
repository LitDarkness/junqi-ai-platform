from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dig(value: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def ci(value: Any) -> tuple[Any, Any]:
    if isinstance(value, list) and len(value) >= 2:
        return value[0], value[1]
    return None, None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iteration_number(agent: str) -> int | None:
    match = re.search(r"(?:bare|search64(?:r128)?)_(\d+)", agent)
    return int(match.group(1)) if match else None


def lineage_reports(root: Path) -> list[tuple[str, Path]]:
    runs = root / "runs"
    groups = [
        ("v3_initial", runs / "open_junqi_search_v3" / "loop"),
        (
            "v3_alternating",
            runs / "open_junqi_search_v3_alternating" / "reports",
        ),
        (
            "v3_scaled",
            runs / "open_junqi_search_v3_scaled_training" / "reports",
        ),
        (
            "v3_strength_continuation",
            runs
            / "open_junqi_search_v3_strength_continuation"
            / "reports",
        ),
    ]
    rows = []
    for lineage, directory in groups:
        for path in sorted(directory.glob("iteration_*.json")):
            if re.fullmatch(r"iteration_\d{6}\.json", path.name):
                rows.append((lineage, path))
    return rows


def external_rows(
    payload: dict[str, Any],
    *,
    iteration: int,
    lineage: str,
    source: str,
) -> list[dict[str, Any]]:
    arena = payload.get("external_arena") or payload.get("post_arena") or {}
    candidates: list[tuple[str, dict[str, Any]]] = []
    direct = arena.get("direct_search64_vs_bare")
    if isinstance(direct, dict):
        candidates.append(("search64_vs_bare", direct))
    for label, result in (arena.get("external") or {}).items():
        if isinstance(result, dict):
            candidates.append((label, result))
    result_rows = []
    for label, result in candidates:
        lower, upper = ci(result.get("ci95"))
        primary = str(result.get("candidate", label.split("_vs_")[0]))
        secondary = str(
            result.get("opponent", label.split("_vs_")[-1])
        )
        result_rows.append(
            {
                "benchmark_class": "per_iteration_deterministic_diagnostic",
                "iteration_context": iteration,
                "primary_agent": primary,
                "secondary_agent": secondary,
                "primary_iteration": iteration,
                "secondary_iteration": None,
                "matchup": label,
                "games": result.get("games"),
                "paired_states": None,
                "trajectory_families": None,
                "wins": result.get("wins"),
                "draws": result.get("draws"),
                "losses": result.get("losses"),
                "score": result.get("score"),
                "state_weighted_score": result.get("score"),
                "family_weighted_score": None,
                "ci95_low": lower,
                "ci95_high": upper,
                "family_ci95_low": None,
                "family_ci95_high": None,
                "repetition_fraction": result.get(
                    "repetition_fraction"
                ),
                "capture_flag_fraction": result.get(
                    "capture_flag_fraction"
                ),
                "average_ply": result.get("average_ply"),
                "unique_trajectories": result.get(
                    "unique_trajectories"
                ),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "search_simulations_per_second": None,
                "statistical_unit": (
                    "deterministic trajectories; not independent games"
                ),
                "diagnostic_only": True,
                "pseudoreplication_warning": result.get(
                    "deterministic_pseudoreplication_warning", True
                ),
                "lineage": lineage,
                "source_path": source,
            }
        )
    return result_rows


def collect_iterations(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    iterations: list[dict[str, Any]] = []
    matchups: list[dict[str, Any]] = []
    champion_path = (
        root
        / "runs"
        / "open_junqi_search_v3_strength_continuation"
        / "championship.json"
    )
    champions: set[int] = set()
    if champion_path.exists():
        champion = load(champion_path)
        champions.update(
            int(item["iteration"]) for item in champion.get("history", [])
        )
        current = champion.get("current") or {}
        if current.get("iteration") is not None:
            champions.add(int(current["iteration"]))
    for lineage, path in lineage_reports(root):
        payload = load(path)
        iteration = int(payload["iteration"])
        data = payload.get("data") or {}
        train = payload.get("fresh_train") or {}
        held = (
            payload.get("held_out")
            or payload.get("distill_after")
            or payload.get("validation")
            or {}
        )
        pair_policy = payload.get("paired_policy") or {}
        pair_wdl = payload.get("paired_wdl") or {}
        kl = payload.get("value_phase_policy_kl") or {}
        pre = payload.get("pre_probe") or {}
        post = payload.get("post_probe") or {}
        chosen = payload.get("chosen_candidate") or {}
        round_summary = payload.get("round_summary") or {}
        reasons = payload.get("reasons") or []
        passed = payload.get("passed")
        if passed is None:
            passed = len(reasons) == 0
        selected_epoch = (
            round_summary.get("selected_fractional_epoch")
            if round_summary
            else chosen.get(
                "fractional_epoch", payload.get("chosen_epoch")
            )
        )
        update = round_summary.get("update")
        if update is None:
            update = (
                "accepted_update"
                if selected_epoch not in (None, 0, 0.0)
                else "safe_no_op"
            )
        term = data.get("terminal_reasons") or {}
        games = data.get("games")
        pre_ci = ci(pair_policy.get("ci95"))
        wdl_ci = ci(pair_wdl.get("ci95"))
        row = {
            "iteration": iteration,
            "lineage": lineage,
            "passed": bool(passed),
            "update": update,
            "selected_epoch": selected_epoch,
            "is_frozen_champion": iteration in champions,
            "games": games,
            "policy_samples": data.get("policy_samples"),
            "unique_states": data.get("unique_states"),
            "unique_state_ratio": data.get("unique_state_ratio"),
            "unique_trajectories": data.get("unique_trajectories"),
            "root_visit_entropy": data.get("root_visit_entropy"),
            "repetition_fraction": data.get("repetition_fraction"),
            "capture_flag_fraction": (
                term.get("FLAG_CAPTURE") / games
                if games and term.get("FLAG_CAPTURE") is not None
                else None
            ),
            "no_legal_move_fraction": (
                term.get("NO_LEGAL_MOVE") / games
                if games and term.get("NO_LEGAL_MOVE") is not None
                else None
            ),
            "average_selfplay_ply": data.get("average_ply"),
            "selfplay_simulations_per_second": data.get(
                "simulations_per_second"
            ),
            "cycle_suffix_samples_removed": dig(
                data, "replay_cycle_cleanup", "cycle_suffix_samples_removed"
            ),
            "cycle_teacher_roots": dig(data, "cycle_teacher", "roots"),
            "cycle_teacher_adjusted": dig(
                data, "cycle_teacher", "adjusted"
            ),
            "train_policy_excess": train.get(
                "policy_excess", train.get("excess")
            ),
            "heldout_policy_excess": held.get(
                "policy_excess", held.get("excess")
            ),
            "heldout_policy_top1": held.get("top1"),
            "heldout_wdl_ce": held.get("wdl_ce", held.get("wdl")),
            "policy_family_classification": pair_policy.get(
                "classification"
            ),
            "policy_family_delta": pair_policy.get("mean_delta"),
            "policy_family_ci_low": pre_ci[0],
            "policy_family_ci_high": pre_ci[1],
            "wdl_family_classification": pair_wdl.get("classification"),
            "wdl_family_delta": pair_wdl.get("mean_delta"),
            "wdl_family_ci_low": wdl_ci[0],
            "wdl_family_ci_high": wdl_ci[1],
            "value_step_peak_kl": kl.get(
                "value_step_peak_kl", kl.get("maximum")
            ),
            "candidate_final_kl": kl.get("candidate_final_kl"),
            "candidate_final_family_kl": kl.get(
                "candidate_final_family_kl"
            ),
            "pre_bare_oracle_regret": pre.get("bare_oracle_regret"),
            "pre_search64_oracle_regret": pre.get(
                "search64_oracle_regret"
            ),
            "post_bare_oracle_regret": post.get("bare_oracle_regret"),
            "post_search64_oracle_regret": post.get(
                "search64_oracle_regret"
            ),
            "post_search64_regret_reduction": post.get(
                "search_gain_regret_fraction"
            ),
            "post_search64_sps": post.get("search64_sps"),
            "post_bare_search_action_agreement": post.get(
                "bare_search_action_agreement"
            ),
            "post_bare_avoidable_exit_fraction": post.get(
                "bare_avoidable_exit_fraction"
            ),
            "post_search64_avoidable_exit_fraction": post.get(
                "search64_avoidable_exit_fraction"
            ),
            "policy_steps": round_summary.get(
                "policy_steps", chosen.get("policy_steps")
            ),
            "value_steps": round_summary.get(
                "value_steps", chosen.get("value_steps")
            ),
            "global_step": round_summary.get("global_step"),
            "model_sha256": round_summary.get("model_sha256"),
            "reasons": " | ".join(map(str, reasons)),
            "source_path": str(path),
        }
        iterations.append(row)
        matchups.extend(
            external_rows(
                payload,
                iteration=iteration,
                lineage=lineage,
                source=str(path),
            )
        )
    iterations.sort(key=lambda row: int(row["iteration"]))
    return iterations, matchups


def benchmark_class(path: Path) -> str:
    text = str(path).lower()
    if "teacher_vs_exact082" in text:
        return "teacher_vs_exact082_formal"
    if "\\championships\\" in text:
        return "sealed_championship"
    if "\\candidate_dev\\" in text:
        return "candidate_dev_continuation"
    if "sealed_continuation_v1" in text:
        return "sealed_iteration50_regression"
    if "checkpoint_benchmark" in text:
        return "checkpoint_crossplay_diagnostic"
    if "pareto_79_80" in text:
        return "pareto_candidate_dev"
    if "open_junqi_search_v3_strength\\continuation_arena" in text:
        return "iteration10_strength_continuation"
    return "other_formal_continuation"


def family_weighted_from_rows(result: dict[str, Any]) -> float | None:
    rows_path = result.get("rows_path")
    if not rows_path:
        return None
    path = Path(rows_path)
    if not path.exists():
        return None
    by_family: dict[int, list[float]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                family = int(row["trajectory_family_id"])
                by_family.setdefault(family, []).append(float(row["score"]))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if not by_family:
        return None
    family_means = [
        sum(scores) / len(scores) for scores in by_family.values()
    ]
    return sum(family_means) / len(family_means)


def oracle_budgeted_performance(
    result: dict[str, Any], contract: dict[str, Any]
) -> tuple[int | None, int | None, float | None]:
    budget = contract.get("oracle_budget_total_root_nodes")
    rows_path = result.get("rows_path")
    suite_path = contract.get("suite")
    elapsed = dig(result, "performance", "elapsed_seconds")
    if not budget or not rows_path or not suite_path or not elapsed:
        return None, None, None
    rows_file = Path(rows_path)
    suite_file = Path(suite_path)
    if not rows_file.exists() or not suite_file.exists():
        return None, None, None
    current_player = {}
    try:
        for line in suite_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            current_player[int(row["suite_id"])] = int(
                row["current_player"]
            )
        calls = 0
        with rows_file.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                plies = int(row["continuation_plies"])
                starts = current_player[int(row["suite_id"])]
                primary = int(row["primary_side"])
                calls += (plies + int(starts == primary)) // 2
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, None, None
    nodes = calls * int(budget)
    return calls, nodes, nodes / float(elapsed)


def collect_matchup_reports(root: Path) -> list[dict[str, Any]]:
    runs = root / "runs"
    prefixes = [
        "open_junqi_search_v3_strength",
        "open_junqi_search_v3_sealed_continuation_v1",
        "open_junqi_search_v3_strength_continuation",
        "open_junqi_search_v3_teacher_vs_exact082",
    ]
    paths: list[Path] = []
    for prefix in prefixes:
        directory = runs / prefix
        if directory.exists():
            paths.extend(directory.rglob("report.json"))
            paths.extend(directory.glob("**/report_states_1024.json"))
    # report.json is authoritative when both aliases are present.
    alias_parents = {
        path.parent
        for path in paths
        if path.name == "report.json"
    }
    paths = [
        path
        for path in paths
        if not (
            path.name == "report_states_1024.json"
            and path.parent in alias_parents
        )
    ]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None, str | None]] = set()
    for path in sorted(set(paths)):
        try:
            payload = load(path)
        except Exception:
            continue
        matchups = payload.get("matchups")
        if not isinstance(matchups, dict):
            continue
        contract = payload.get("contract") or {}
        for label, result in matchups.items():
            if not isinstance(result, dict):
                continue
            primary = str(
                result.get("primary_agent")
                or result.get("candidate")
                or label.split("_vs_")[0]
            )
            secondary = str(
                result.get("secondary_agent")
                or result.get("opponent")
                or label.split("_vs_")[-1]
            )
            games = result.get("games")
            key = (str(path.resolve()), label, games, result.get("score"))
            if key in seen:
                continue
            seen.add(key)
            score_ci = ci(
                result.get("paired_state_ci95") or result.get("ci95")
            )
            family_ci = ci(result.get("family_ci95"))
            perf = result.get("performance") or {}
            source_class = benchmark_class(path)
            elapsed_seconds = perf.get(
                "elapsed_seconds", result.get("elapsed_seconds")
            )
            oracle_calls, oracle_nodes, oracle_nps = (
                oracle_budgeted_performance(result, contract)
            )
            rows.append(
                {
                    "benchmark_class": source_class,
                    "iteration_context": (
                        iteration_number(primary)
                        or iteration_number(secondary)
                    ),
                    "primary_agent": primary,
                    "secondary_agent": secondary,
                    "primary_iteration": iteration_number(primary),
                    "secondary_iteration": iteration_number(secondary),
                    "matchup": label,
                    "games": games,
                    "paired_states": result.get("paired_states"),
                    "trajectory_families": result.get(
                        "trajectory_families"
                    ),
                    "wins": result.get("wins"),
                    "draws": result.get("draws"),
                    "losses": result.get("losses"),
                    "score": result.get("score"),
                    "state_weighted_score": result.get(
                        "state_weighted_score", result.get("score")
                    ),
                    "family_weighted_score": result.get(
                        "family_weighted_mean_score"
                    )
                    if result.get("family_weighted_mean_score") is not None
                    else family_weighted_from_rows(result),
                    "ci95_low": score_ci[0],
                    "ci95_high": score_ci[1],
                    "family_ci95_low": family_ci[0],
                    "family_ci95_high": family_ci[1],
                    "repetition_fraction": result.get(
                        "repetition_fraction"
                    ),
                    "capture_flag_fraction": result.get(
                        "capture_flag_fraction"
                    ),
                    "average_ply": result.get(
                        "average_continuation_ply",
                        result.get("average_ply"),
                    ),
                    "unique_trajectories": result.get(
                        "unique_continuation_trajectories",
                        result.get("unique_trajectories"),
                    ),
                    "elapsed_seconds": elapsed_seconds,
                    "games_per_second": (
                        float(games) / float(elapsed_seconds)
                        if games and elapsed_seconds
                        else None
                    ),
                    "search_simulations_per_second": perf.get(
                        "search64_simulations_per_second"
                    ),
                    "oracle_root_calls": oracle_calls,
                    "oracle_budgeted_root_nodes": oracle_nodes,
                    "oracle_budgeted_nodes_per_wall_second": oracle_nps,
                    "statistical_unit": contract.get(
                        "statistical_unit",
                        "paired suite state / trajectory family",
                    ),
                    "diagnostic_only": source_class
                    in {
                        "candidate_dev_continuation",
                        "pareto_candidate_dev",
                        "checkpoint_crossplay_diagnostic",
                    },
                    "pseudoreplication_warning": result.get(
                        "deterministic_pseudoreplication_warning", False
                    ),
                    "suite_sha256": contract.get(
                        "suite_sha256", contract.get("suite_hash")
                    ),
                    "source_path": str(path),
                }
            )
    return rows


def collect_state_probes(root: Path) -> list[dict[str, Any]]:
    path = (
        root
        / "runs"
        / "open_junqi_search_v3_strength"
        / "evaluation"
        / "report.json"
    )
    if not path.exists():
        return []
    payload = load(path)
    network = payload.get("network") or {}
    rows = [
        {
            "evaluation": "bare_vs_search64",
            "reference": "same supervised iteration10 network",
            "states": network.get("states"),
            "bare_regret": None,
            "search64_regret": None,
            "regret_reduction": None,
            "bare_top1_agreement": None,
            "search64_top1_agreement": None,
            "bare_search_agreement": network.get(
                "bare_search_agreement"
            ),
            "search64_simulations_per_second": network.get(
                "search64_simulations_per_second"
            ),
            "source_path": str(path),
        }
    ]
    for budget, result in (payload.get("oracle") or {}).items():
        overall = result.get("overall") or {}
        rows.append(
            {
                "evaluation": f"oracle_{budget}",
                "reference": result.get("reference"),
                "states": overall.get("states"),
                "bare_regret": dig(overall, "bare", "mean_regret"),
                "search64_regret": dig(
                    overall, "search64", "mean_regret"
                ),
                "regret_reduction": overall.get(
                    "search64_regret_reduction"
                ),
                "bare_top1_agreement": dig(
                    overall, "bare", "top1_agreement"
                ),
                "search64_top1_agreement": dig(
                    overall, "search64", "top1_agreement"
                ),
                "bare_search_agreement": network.get(
                    "bare_search_agreement"
                ),
                "search64_simulations_per_second": network.get(
                    "search64_simulations_per_second"
                ),
                "source_path": str(path),
            }
        )
    exact = payload.get("exact082") or {}
    rows.append(
        {
            "evaluation": "exact082_action_agreement",
            "reference": exact.get("reference"),
            "states": exact.get("states"),
            "bare_regret": None,
            "search64_regret": None,
            "regret_reduction": None,
            "bare_top1_agreement": exact.get("bare_top1_agreement"),
            "search64_top1_agreement": exact.get(
                "search64_top1_agreement"
            ),
            "bare_search_agreement": network.get(
                "bare_search_agreement"
            ),
            "search64_simulations_per_second": network.get(
                "search64_simulations_per_second"
            ),
            "source_path": str(path),
        }
    )
    return rows


def collect_initial_search_gain(root: Path) -> list[dict[str, Any]]:
    path = (
        root
        / "runs"
        / "open_junqi_search_v3"
        / "search_gain_report.json"
    )
    if not path.exists():
        return []
    payload = load(path)
    rows = []
    for budget in ("bare", "8", "16", "32", "64", "128"):
        result = dig(payload, "probe_variants", budget, default={})
        perf = dig(payload, "performance", budget, default={})
        rows.append(
            {
                "network_stage": "supervised_warm_start",
                "search_simulations": 0 if budget == "bare" else int(budget),
                "probe_states": payload.get("probe_count"),
                "oracle_regret_mean": result.get("oracle_regret_mean"),
                "oracle_regret_median": result.get(
                    "oracle_regret_median"
                ),
                "root_entropy_mean": result.get("root_entropy_mean"),
                "repetition_action_fraction": result.get(
                    "repetition_action_fraction"
                ),
                "avoidable_exit_fraction": dig(
                    result,
                    "exit_fraction_by_class",
                    "avoidable_repetition",
                ),
                "simulations_per_second": perf.get(
                    "simulations_per_second"
                ),
                "mean_inference_batch": perf.get(
                    "mean_inference_batch"
                ),
                "source_path": str(path),
            }
        )
    return rows


def collect_supervised(root: Path) -> list[dict[str, Any]]:
    path = (
        root
        / "runs"
        / "open_junqi_search_v3"
        / "supervised_report.json"
    )
    if not path.exists():
        return []
    payload = load(path)
    rows = []
    for split in ("train", "validation", "test"):
        result = payload.get(split) or {}
        rows.append(
            {
                "stage": "selected_supervised_warm_start",
                "split": split,
                "selected_tau": payload.get("selected_tau"),
                "samples": result.get("samples"),
                "policy_ce": result.get("policy"),
                "policy_excess": result.get("excess"),
                "uniform_policy_ce": result.get("uniform"),
                "top1": result.get("top1"),
                "wdl_ce": result.get("wdl"),
                "wdl_accuracy": result.get("wdl_accuracy"),
                "illegal_probability": result.get(
                    "illegal_probability"
                ),
                "source_path": str(path),
            }
        )
    overfit = payload.get("overfit") or {}
    policy = overfit.get("policy_final") or {}
    value = overfit.get("value_final") or {}
    rows.append(
        {
            "stage": "2048_single_batch_overfit",
            "split": "same_batch",
            "selected_tau": overfit.get("tau"),
            "samples": overfit.get("samples"),
            "policy_ce": policy.get("policy"),
            "policy_excess": policy.get("excess"),
            "uniform_policy_ce": None,
            "top1": policy.get("top1"),
            "wdl_ce": value.get("wdl"),
            "wdl_accuracy": value.get("wdl_accuracy"),
            "illegal_probability": policy.get("illegal_probability"),
            "source_path": str(path),
        }
    )
    return rows


def collect_champions(root: Path) -> list[dict[str, Any]]:
    path = (
        root
        / "runs"
        / "open_junqi_search_v3_strength_continuation"
        / "championship.json"
    )
    if not path.exists():
        return []
    payload = load(path)
    entries = list(payload.get("history") or [])
    current = payload.get("current") or {}
    if current:
        entries.append(current)
    rows = []
    seen = set()
    for entry in entries:
        iteration = int(entry["iteration"])
        if iteration in seen:
            continue
        seen.add(iteration)
        rows.append(
            {
                "iteration": iteration,
                "status": (
                    "current"
                    if iteration == int(current.get("iteration", -1))
                    else "historical"
                ),
                "source": entry.get("source"),
                "replaced_by": entry.get("replaced_by"),
                "checkpoint_sha256": entry.get("checkpoint_sha256"),
                "checkpoint": entry.get("checkpoint"),
                "decision_report": entry.get(
                    "championship_decision"
                ),
                "source_path": str(path),
            }
        )
    return sorted(rows, key=lambda row: row["iteration"])


def make_summary(
    iterations: list[dict[str, Any]],
    matchups: list[dict[str, Any]],
    champions: list[dict[str, Any]],
) -> dict[str, Any]:
    formal = [
        row
        for row in matchups
        if not row.get("diagnostic_only")
        and not row.get("pseudoreplication_warning")
    ]
    teacher = [
        row
        for row in formal
        if row["benchmark_class"] == "teacher_vs_exact082_formal"
    ]
    latest = iterations[-1] if iterations else {}
    first = iterations[0] if iterations else {}
    return {
        "schema": "open-junqi-search-v3-benchmark-ledger-v1",
        "iteration_range": [
            first.get("iteration"),
            latest.get("iteration"),
        ],
        "iteration_rows": len(iterations),
        "accepted_updates": sum(
            row.get("update") == "accepted_update" for row in iterations
        ),
        "safe_no_ops": sum(
            row.get("update") == "safe_no_op" for row in iterations
        ),
        "champions": [row["iteration"] for row in champions],
        "formal_matchup_rows": len(formal),
        "all_matchup_rows": len(matchups),
        "teacher_vs_exact082": teacher[-1] if teacher else None,
        "latest_iteration": {
            key: latest.get(key)
            for key in (
                "iteration",
                "update",
                "heldout_policy_excess",
                "heldout_wdl_ce",
                "post_bare_oracle_regret",
                "post_search64_oracle_regret",
                "post_search64_regret_reduction",
                "repetition_fraction",
                "selfplay_simulations_per_second",
            )
        },
        "benchmark_class_counts": dict(
            Counter(row["benchmark_class"] for row in matchups)
        ),
    }


def fmt_pct(value: Any) -> str:
    return "—" if value is None or value == "" else f"{100 * float(value):.2f}%"


def fmt_num(value: Any, digits: int = 4) -> str:
    return "—" if value is None or value == "" else f"{float(value):.{digits}f}"


def write_markdown(
    path: Path,
    *,
    iterations: list[dict[str, Any]],
    matchups: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    initial_search: list[dict[str, Any]],
    supervised: list[dict[str, Any]],
    champions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    latest = iterations[-1]
    formal = [
        row
        for row in matchups
        if not row.get("pseudoreplication_warning")
        and row["benchmark_class"]
        != "per_iteration_deterministic_diagnostic"
    ]
    teacher = [
        row
        for row in formal
        if row["benchmark_class"] == "teacher_vs_exact082_formal"
    ]
    selected_matchups = {
        row["matchup"]: row
        for row in formal
        if row["benchmark_class"]
        in {
            "iteration10_strength_continuation",
            "sealed_iteration50_regression",
            "sealed_championship",
        }
    }
    lines = [
        "# Open Junqi Search-v3 Benchmark 汇总",
        "",
        f"快照覆盖 Iteration 1–{latest['iteration']}，共 "
        f"{summary['accepted_updates']} 次 accepted update、"
        f"{summary['safe_no_ops']} 次 safe no-op。Frozen champion："
        + " → ".join(str(row["iteration"]) for row in champions)
        + "。",
        "",
        "## 1. 初始监督热启动",
        "",
        "| 阶段 | Split | Policy excess | Top-1 | WDL CE | 非法动作概率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in supervised:
        lines.append(
            f"| {row['stage']} | {row['split']} | "
            f"{fmt_num(row.get('policy_excess'))} | "
            f"{fmt_pct(row.get('top1'))} | "
            f"{fmt_num(row.get('wdl_ce'))} | "
            f"{fmt_pct(row.get('illegal_probability'))} |"
        )
    lines += [
        "",
        "初始 Policy 标签来自 `supplement_wide_quiescence`：普通状态 "
        "2,000 总根节点，重要/循环入口状态 8,000，32,000 仅用于 probe；"
        "exact082 是轨迹对手之一，不是 Policy 标签教师。",
        "",
        "## 2. 初始裸网络与搜索收益",
        "",
        "| PUCT simulations | Oracle regret | 重复动作率 | 可避免循环退出率 | sim/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in initial_search:
        lines.append(
            f"| {row['search_simulations']} | "
            f"{fmt_num(row.get('oracle_regret_mean'))} | "
            f"{fmt_pct(row.get('repetition_action_fraction'))} | "
            f"{fmt_pct(row.get('avoidable_exit_fraction'))} | "
            f"{fmt_num(row.get('simulations_per_second'), 0)} |"
        )
    lines += [
        "",
        "## 3. 1024-state 静态参照（Iteration 10）",
        "",
        "| 参照 | Bare regret/一致率 | Search64 regret/一致率 | 相对变化 |",
        "|---|---:|---:|---:|",
    ]
    for row in probes:
        bare = (
            row.get("bare_regret")
            if row.get("bare_regret") is not None
            else row.get("bare_top1_agreement")
        )
        search = (
            row.get("search64_regret")
            if row.get("search64_regret") is not None
            else row.get("search64_top1_agreement")
        )
        as_pct = row.get("bare_regret") is None
        lines.append(
            f"| {row['reference']} | "
            f"{fmt_pct(bare) if as_pct else fmt_num(bare)} | "
            f"{fmt_pct(search) if as_pct else fmt_num(search)} | "
            f"{fmt_pct(row.get('regret_reduction'))} |"
        )
    lines += [
        "",
        "静态 Oracle regret 与完整续弈胜负不是同一个指标；Iteration 10 "
        "的 Search64 在深 Oracle 静态 regret 上并未优于 bare，但在同一 "
        "1024-state 完整续弈中明显更强。",
        "",
        "## 4. 关键正式完整续弈",
        "",
        "| Matchup | W/D/L | Score | Family score | Family 95% CI | 重复率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    highlight_labels = [
        "bare_vs_exact082",
        "search64_vs_exact082",
        "search64_vs_bare",
        "bare_50_vs_bare_10",
        "search64_50_vs_search64_10",
        "bare_83_vs_bare_50",
        "search64_83_vs_search64_50",
        "bare_88_vs_bare_83",
        "search64_88_vs_search64_83",
        "bare_92_vs_bare_88",
        "search64_92_vs_search64_88",
        "bare_100_vs_bare_92",
        "search64_100_vs_search64_92",
        "bare_100_vs_exact082",
        "search64_100_vs_exact082",
        "search64_100_vs_bare_100",
    ]
    for label in highlight_labels:
        row = selected_matchups.get(label)
        if not row:
            continue
        lines.append(
            f"| {label} | {row.get('wins')}/{row.get('draws')}/"
            f"{row.get('losses')} | {fmt_pct(row.get('score'))} | "
            f"{fmt_pct(row.get('family_weighted_score'))} | "
            f"[{fmt_pct(row.get('family_ci95_low'))}, "
            f"{fmt_pct(row.get('family_ci95_high'))}] | "
            f"{fmt_pct(row.get('repetition_fraction'))} |"
        )
    lines += [
        "",
        "## 5. 热启动教师 Heuristic vs exact082",
        "",
    ]
    if teacher:
        row = teacher[-1]
        lines += [
            f"- 对局：`{row['primary_agent']} vs {row['secondary_agent']}`，"
            f"{row['paired_states']} 个状态逐状态换边，共 {row['games']} 局。",
            f"- W/D/L = {row['wins']}/{row['draws']}/{row['losses']}；"
            f"state-weighted score = {fmt_pct(row['state_weighted_score'])}。",
            f"- family-weighted score = {fmt_pct(row['family_weighted_score'])}；"
            f"family 95% CI = [{fmt_pct(row['family_ci95_low'])}, "
            f"{fmt_pct(row['family_ci95_high'])}]。",
            f"- 重复率 {fmt_pct(row['repetition_fraction'])}，平均续弈 "
            f"{fmt_num(row['average_ply'], 2)} ply。",
            f"- 约 {row.get('oracle_root_calls')} 次 Oracle 根调用、"
            f"{row.get('oracle_budgeted_root_nodes')} 预算节点；包含 exact082 "
            f"耗时的总墙钟吞吐约 "
            f"{fmt_num(row.get('oracle_budgeted_nodes_per_wall_second'), 0)} "
            "budgeted node/s。",
        ]
    else:
        lines.append("- 正式评测尚未完成。")
    lines += [
        "",
        "## 6. 使用限制",
        "",
        "- 不同 sealed suite 的绝对得分不可直接作同分布时间序列；同一报告内的 "
        "candidate/champion paired delta 才是最强因果证据。",
        "- 每轮日志中的 128 局通常只有 2 条唯一确定性轨迹，已单独归入“每轮诊断”，"
        "不得按 128 个独立样本解释。",
        "- exact082 只返回动作，不提供逐动作校准 Q；对它的静态指标只能报告 "
        "top-1 agreement，真实棋力以完整续弈为准。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    iterations, daily = collect_iterations(root)
    formal_matchups = collect_matchup_reports(root)
    matchups = formal_matchups + daily
    probes = collect_state_probes(root)
    initial_search = collect_initial_search_gain(root)
    supervised = collect_supervised(root)
    champions = collect_champions(root)
    summary = make_summary(iterations, matchups, champions)
    ledger = {
        "summary": summary,
        "iterations": iterations,
        "matchups": matchups,
        "state_probes": probes,
        "initial_search_gain": initial_search,
        "supervised_warm_start": supervised,
        "champions": champions,
    }

    write_csv(output / "iteration_history.csv", iterations)
    write_csv(output / "matchup_results.csv", matchups)
    write_csv(output / "state_probe_results.csv", probes)
    write_csv(output / "initial_search_gain.csv", initial_search)
    write_csv(output / "supervised_warm_start.csv", supervised)
    write_csv(output / "champion_history.csv", champions)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "ledger_data.json").write_text(
        json.dumps(ledger, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(
        output / "benchmark_summary.md",
        iterations=iterations,
        matchups=matchups,
        probes=probes,
        initial_search=initial_search,
        supervised=supervised,
        champions=champions,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
