"""
model.py
Core NHL game-level model: moneyline, puck line (-1.5/+1.5), and totals.
Same statistical backbone as the MLB app (Poisson/negative-binomial on goals-for/against),
with two NHL-specific adjustments layered on top:
  1. Goalie adjustment — shifts expected goals against based on the starting goalie
     vs. the team's baseline save percentage.
  2. Empty-net correction on the puck line — raw Poisson understates 2-goal margins
     because it doesn't know about empty-net goals in the final ~2 minutes.
"""

from dataclasses import dataclass
from scipy.stats import poisson

HOME_ICE_GOAL_FACTOR = 1.06  # home teams score ~6% more than a neutral-site average; tune from backtests
MAX_GOALS = 10               # truncate the scoring matrix here; NHL games essentially never exceed this
EMPTY_NET_1_GOAL_TO_2_GOAL_RATE = 0.22  # share of 1-goal-margin regulation wins that become 2-goal via empty net
                                         # starting estimate — replace with your own backtested figure


@dataclass
class GoalieAdjustment:
    goalie_save_pct: float
    team_baseline_save_pct: float

    def defense_multiplier(self) -> float:
        """
        >1.0 means this goalie is worse than the team's baseline (opponent should score MORE).
        <1.0 means this goalie is better than baseline (opponent should score LESS).
        """
        if not self.goalie_save_pct or not self.team_baseline_save_pct:
            return 1.0
        # Convert the save-pct gap into a goals-against multiplier.
        # e.g. goalie at .910 vs team baseline .905 -> saves ~0.5% more shots -> ~5% fewer goals against
        # (scaled: shots against per game ~30, so 1pp of save% ≈ ~0.3 goals)
        save_pct_gap = self.team_baseline_save_pct - self.goalie_save_pct
        return 1.0 + (save_pct_gap * 10)  # tune this scaling factor against real results, same as MLB calibration


def expected_goals(home_abbrev: str, away_abbrev: str, strengths: dict,
                    home_goalie_adj: GoalieAdjustment = None,
                    away_goalie_adj: GoalieAdjustment = None) -> tuple[float, float]:
    """Returns (home_expected_goals, away_expected_goals)."""
    league_avg = strengths["_league_avg"]
    home = strengths[home_abbrev]
    away = strengths[away_abbrev]

    home_exp = league_avg * home["attack"] * away["defense"] * HOME_ICE_GOAL_FACTOR
    away_exp = league_avg * away["attack"] * home["defense"]

    # Goalie adjustment acts on the OPPONENT's expected goals (a good away goalie suppresses home_exp)
    if away_goalie_adj:
        home_exp *= away_goalie_adj.defense_multiplier()
    if home_goalie_adj:
        away_exp *= home_goalie_adj.defense_multiplier()

    return home_exp, away_exp


def score_matrix(home_exp: float, away_exp: float) -> list[list[float]]:
    """P(home scores i, away scores j) for i, j in 0..MAX_GOALS, independent Poisson."""
    home_pmf = [poisson.pmf(i, home_exp) for i in range(MAX_GOALS + 1)]
    away_pmf = [poisson.pmf(j, away_exp) for j in range(MAX_GOALS + 1)]
    return [[home_pmf[i] * away_pmf[j] for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]


def moneyline_probs(matrix: list[list[float]], home_exp: float, away_exp: float) -> dict:
    """
    NHL has no ties. Regulation ties go to OT/shootout, split roughly by relative team strength
    (a coin flip weighted slightly by which team was expected to be the stronger side).
    """
    home_reg_win = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > j)
    away_reg_win = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if j > i)
    reg_tie = sum(matrix[i][i] for i in range(MAX_GOALS + 1))

    # Split the tied-game probability using relative expected-goals strength as a proxy for OT/SO edge
    total_exp = home_exp + away_exp
    home_ot_share = (home_exp / total_exp) if total_exp > 0 else 0.5

    home_win = home_reg_win + reg_tie * home_ot_share
    away_win = away_reg_win + reg_tie * (1 - home_ot_share)

    return {"home_win_prob": home_win, "away_win_prob": away_win, "reg_tie_prob": reg_tie}


def puck_line_probs(matrix: list[list[float]]) -> dict:
    """
    Standard -1.5/+1.5 puck line, WITHOUT empty-net correction.
    home_covers_-1.5 = P(home wins by 2+)
    away_covers_+1.5 = P(away loses by 1, or wins)
    """
    home_by_2plus = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i - j >= 2)
    home_by_1 = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i - j == 1)
    away_by_2plus = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if j - i >= 2)
    away_by_1 = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if j - i == 1)
    tie = sum(matrix[i][i] for i in range(MAX_GOALS + 1))

    return {
        "home_-1.5_raw": home_by_2plus,
        "away_+1.5_raw": away_by_2plus + away_by_1 + home_by_1 + tie,  # away covers unless home wins by 2+
        "home_by_1": home_by_1,
        "away_by_1": away_by_1,
        "tie": tie,
    }


def puck_line_probs_with_empty_net(matrix: list[list[float]]) -> dict:
    """
    Corrects the raw puck line for empty-net goals: a real share of 1-goal regulation wins
    become 2-goal final margins once the trailing team pulls its goalie. This moves probability
    from 'home_by_1' into 'home_-1.5_raw', which is exactly the gap the MLB app's plausibility
    check was designed to catch on the run line — same failure mode, different sport.
    """
    raw = puck_line_probs(matrix)

    shifted_from_home_1 = raw["home_by_1"] * EMPTY_NET_1_GOAL_TO_2_GOAL_RATE
    shifted_from_away_1 = raw["away_by_1"] * EMPTY_NET_1_GOAL_TO_2_GOAL_RATE

    home_covers = raw["home_-1.5_raw"] + shifted_from_home_1
    away_covers = 1 - home_covers  # two-outcome market once you fold in the shift

    return {
        "home_-1.5": home_covers,
        "away_+1.5": away_covers,
    }


def totals_probs(matrix: list[list[float]], line: float) -> dict:
    """P(over) / P(under) for a given total, e.g. line=6.0 for O/U 6."""
    over = sum(matrix[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i + j > line)
    under = 1 - over
    return {"over": over, "under": under}


def run_game_model(home_abbrev: str, away_abbrev: str, strengths: dict,
                    total_line: float = 6.0,
                    home_goalie_adj: GoalieAdjustment = None,
                    away_goalie_adj: GoalieAdjustment = None) -> dict:
    """Single entry point — mirrors the MLB app's per-game model call."""
    home_exp, away_exp = expected_goals(home_abbrev, away_abbrev, strengths, home_goalie_adj, away_goalie_adj)
    matrix = score_matrix(home_exp, away_exp)

    return {
        "home_exp_goals": round(home_exp, 2),
        "away_exp_goals": round(away_exp, 2),
        "moneyline": moneyline_probs(matrix, home_exp, away_exp),
        "puck_line_raw": puck_line_probs(matrix),
        "puck_line": puck_line_probs_with_empty_net(matrix),
        "totals": totals_probs(matrix, total_line),
    }
