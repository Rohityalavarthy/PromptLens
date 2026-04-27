from promptlens import SaliencyReport, CompressionResult

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
ORANGE = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
GRAY   = "\033[90m"
CYAN   = "\033[96m"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    color = GREEN if score < 0.15 else (ORANGE if score < 0.5 else RED)
    return color + "█" * filled + GRAY + "░" * (width - filled) + RESET


def print_saliency_report(report: SaliencyReport, file: str = "") -> None:
    print()
    if file:
        print(f"{BOLD}📋 PromptLens Analysis — {file}{RESET}")
    print(f"{GRAY}{'─' * 60}{RESET}")
    print(f"  Phrases analysed : {BOLD}{len(report.phrases)}{RESET}")
    print(f"  Token estimate   : {BOLD}{report.token_count}{RESET}")
    print(f"  Test inputs used : {BOLD}{report.test_inputs_used}{RESET}")
    print(f"  Confidence       : {BOLD}{report.confidence:.2f}{RESET}")
    print(f"  Est. redundancy  : {BOLD}{report.redundancy_fraction * 100:.0f}%{RESET}")
    print()

    print(f"  {BOLD}{'PHRASE':<50} {'SCORE':>6}  {'IMPACT'}{RESET}")
    print(f"  {GRAY}{'─' * 50} {'─' * 6}  {'─' * 22}{RESET}")

    for score in sorted(report.scores, key=lambda s: s.score, reverse=True):
        phrase_preview = score.phrase.text[:48] + ".." if len(score.phrase.text) > 48 else score.phrase.text
        score_color = GREEN if score.score < 0.15 else (ORANGE if score.score < 0.5 else RED)
        label = f"  {phrase_preview:<50} {score_color}{score.score:>6.2f}{RESET}  {_bar(score.score)}"
        print(label)

    print()
    print(f"  {BOLD}Candidates for compression:{RESET} "
          f"{RED}{len([s for s in report.scores if s.disposition == 'remove'])} phrases{RESET} "
          f"/ {report.compression_candidate_tokens} tokens")
    print()


def print_audit_summary(discoveries: list, reports: dict) -> None:
    total_tokens = sum(r.token_count for r in reports.values())
    candidate_tokens = sum(r.compression_candidate_tokens for r in reports.values())

    print()
    print(f"{BOLD}{CYAN}🔍 PromptLens Agent — Audit Complete{RESET}")
    print(f"{GRAY}{'═' * 60}{RESET}")
    print(f"  Prompts found    : {BOLD}{len(discoveries)}{RESET}")
    print(f"  Total tokens     : {BOLD}{total_tokens:,}{RESET}")
    if total_tokens > 0:
        print(f"  Candidate tokens : {RED}{BOLD}{candidate_tokens:,}{RESET} ({candidate_tokens/total_tokens*100:.0f}% of total)")
    print()

    print(f"  {BOLD}{'FILE':<45} {'TOKENS':>7}  {'REDUNDANCY':>10}{RESET}")
    print(f"  {GRAY}{'─' * 45} {'─' * 7}  {'─' * 10}{RESET}")
    for path, report in sorted(reports.items(), key=lambda x: x[1].redundancy_fraction, reverse=True):
        fname = path[-43:] if len(path) > 43 else path
        pct = f"{report.redundancy_fraction * 100:.0f}%"
        color = RED if report.redundancy_fraction > 0.4 else (ORANGE if report.redundancy_fraction > 0.2 else GREEN)
        print(f"  {fname:<45} {report.token_count:>7,}  {color}{pct:>10}{RESET}")

    print()
    print(f"  Run {CYAN}promptlens compress --file <path>{RESET} to compress a specific prompt.")
    print()


def print_compression_result(result: CompressionResult) -> None:
    print()
    print(f"{BOLD}{CYAN}✂  Compression Result{RESET}")
    print(f"{GRAY}{'─' * 60}{RESET}")

    status = f"{GREEN}✓ PASSED{RESET}" if result.validation_passed else f"{RED}✗ FAILED{RESET}"
    print(f"  Validation       : {status}")
    print(f"  Max divergence   : {result.worst_case_divergence:.3f}")
    print(f"  Original tokens  : {result.original_tokens:,}")
    print(f"  Compressed tokens: {result.compressed_tokens:,}")
    print(f"  Token reduction  : {BOLD}{GREEN}{result.token_delta:,} tokens "
          f"({result.token_delta / result.original_tokens * 100:.0f}%){RESET}")

    print()
    print(f"  {BOLD}Changes:{RESET}")
    for entry in result.diff:
        if entry["action"] == "keep":
            continue
        icon = {"remove": "🗑 ", "rewrite": "✏️ ", "merge": "⊕ "}.get(entry["action"], "  ")
        preview = entry["original"][:50] + ".." if len(entry["original"]) > 50 else entry["original"]
        if entry["action"] == "remove":
            print(f"  {icon} {RED}{preview}{RESET}")
        else:
            result_preview = entry["result"][:40] + ".." if len(entry["result"]) > 40 else entry["result"]
            print(f"  {icon} {ORANGE}{preview}{RESET} → {GREEN}{result_preview}{RESET}")

    print()
    print(f"  Compressed prompt written to: {CYAN}<file>.suggested{RESET}")
    print()
