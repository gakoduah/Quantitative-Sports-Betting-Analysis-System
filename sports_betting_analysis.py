#!/usr/bin/env python3
"""
===========================================================================
  Quantitative Sports Betting Analysis System
  -------------------------------------------
  A computational framework implementing the mathematical models
  presented in the U.S. Sports Betting Industry presentation.

  Modules:
    1. Odds Conversion & Implied Probability Engine
    2. Expected Value (EV) Calculator
    3. Elo Rating System for NBA Teams
    4. Monte Carlo Season Simulator
    5. Parlay Risk Analyzer
    6. Market Efficiency & Vig Analysis
    7. Kelly Criterion Position Sizing
    8. Backtest & Validation Framework

  Authors: Group 2 — Godfred A. Koduah, Charles Challender, Minh Do
  Date:    April 2026
  Course:  Quantitative Finance — Florida State University
===========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ── Output directory ──
OUT = "/home/claude/project_outputs"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  MODULE 1 — ODDS CONVERSION & IMPLIED PROBABILITY
# ═══════════════════════════════════════════════════════════════════

class OddsConverter:
    """Convert between American, decimal, fractional odds and
    extract implied probabilities with vig decomposition."""

    @staticmethod
    def american_to_implied(odds: float) -> float:
        """American odds → raw implied probability (includes vig)."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        else:
            return 100.0 / (odds + 100.0)

    @staticmethod
    def american_to_decimal(odds: float) -> float:
        if odds < 0:
            return 1.0 + 100.0 / abs(odds)
        else:
            return 1.0 + odds / 100.0

    @staticmethod
    def decimal_to_american(dec: float) -> float:
        if dec >= 2.0:
            return round((dec - 1.0) * 100.0, 1)
        else:
            return round(-100.0 / (dec - 1.0), 1)

    @staticmethod
    def remove_vig(prob_a: float, prob_b: float):
        """Remove the overround (vig) to get true probabilities."""
        total = prob_a + prob_b
        return prob_a / total, prob_b / total

    @staticmethod
    def compute_vig(odds_a: float, odds_b: float) -> float:
        """Compute the book's margin (vig) from a two-way market."""
        p_a = OddsConverter.american_to_implied(odds_a)
        p_b = OddsConverter.american_to_implied(odds_b)
        return (p_a + p_b - 1.0) * 100.0  # percentage overround


# ═══════════════════════════════════════════════════════════════════
#  MODULE 2 — EXPECTED VALUE CALCULATOR
# ═══════════════════════════════════════════════════════════════════

class EVCalculator:
    """Compute expected value and identify +EV opportunities."""

    @staticmethod
    def expected_value(true_prob: float, american_odds: float,
                       stake: float = 100.0) -> float:
        """EV = p * payout - (1-p) * stake"""
        decimal = OddsConverter.american_to_decimal(american_odds)
        payout = stake * decimal
        ev = true_prob * payout - (1.0 - true_prob) * stake
        # Subtract stake from win branch since payout includes stake
        # Actually: EV = p*(decimal-1)*stake - (1-p)*stake
        ev = true_prob * (decimal - 1.0) * stake - (1.0 - true_prob) * stake
        return ev

    @staticmethod
    def edge(true_prob: float, implied_prob: float) -> float:
        """Percentage edge = true_prob - implied_prob."""
        return (true_prob - implied_prob) * 100.0

    @staticmethod
    def breakeven_prob(american_odds: float) -> float:
        """Minimum win rate to break even at given odds."""
        return OddsConverter.american_to_implied(american_odds)


# ═══════════════════════════════════════════════════════════════════
#  MODULE 3 — ELO RATING SYSTEM
# ═══════════════════════════════════════════════════════════════════

class EloSystem:
    """
    Elo rating system for NBA teams.

    E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
    R'_A = R_A + K * (S_A - E(A))

    With home-court advantage adjustment.
    """

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0,
                 initial_rating: float = 1500.0, season_reversion: float = 0.75):
        self.K = k_factor
        self.HOME_ADV = home_advantage
        self.INIT = initial_rating
        self.REVERT = season_reversion
        self.ratings = {}

    def _get_rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.INIT
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float,
                       home_a: bool = False) -> float:
        """Expected score (win probability) for team A."""
        adj_a = rating_a + (self.HOME_ADV if home_a else 0)
        return 1.0 / (1.0 + 10.0 ** ((rating_b - adj_a) / 400.0))

    def win_probability(self, team_a: str, team_b: str,
                        home_a: bool = False) -> float:
        """Win probability for team_a vs team_b."""
        r_a = self._get_rating(team_a)
        r_b = self._get_rating(team_b)
        return self.expected_score(r_a, r_b, home_a)

    def update(self, team_a: str, team_b: str, score_a: float,
               home_a: bool = False, mov: float = 1.0):
        """
        Update ratings after a game.
        score_a: 1.0 for win, 0.0 for loss
        mov: margin of victory multiplier (optional)
        """
        r_a = self._get_rating(team_a)
        r_b = self._get_rating(team_b)
        e_a = self.expected_score(r_a, r_b, home_a)
        e_b = 1.0 - e_a
        score_b = 1.0 - score_a

        # MOV-adjusted K factor
        k_adj = self.K * np.log(max(mov, 1) + 1)

        self.ratings[team_a] = r_a + k_adj * (score_a - e_a)
        self.ratings[team_b] = r_b + k_adj * (score_b - e_b)

    def season_reset(self):
        """Revert ratings toward mean between seasons."""
        for team in self.ratings:
            self.ratings[team] = (self.REVERT * self.ratings[team] +
                                  (1.0 - self.REVERT) * self.INIT)

    def get_rankings(self):
        """Return teams sorted by rating (descending)."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════════
#  MODULE 4 — MONTE CARLO SEASON SIMULATOR
# ═══════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Simulate an NBA season N times using Elo-based win probabilities.
    Produces distributions of win totals, playoff probabilities, etc.
    """

    def __init__(self, elo_system: EloSystem, n_simulations: int = 10000):
        self.elo = elo_system
        self.N = n_simulations

    def simulate_game(self, team_a: str, team_b: str,
                      home_a: bool = False) -> str:
        """Simulate a single game, return winner."""
        p_a = self.elo.win_probability(team_a, team_b, home_a)
        return team_a if np.random.random() < p_a else team_b

    def simulate_season(self, schedule: list) -> dict:
        """
        Simulate a full season.
        schedule: list of (home_team, away_team) tuples
        Returns: {team: wins}
        """
        wins = defaultdict(int)
        for home, away in schedule:
            winner = self.simulate_game(home, away, home_a=True)
            wins[winner] += 1
        return dict(wins)

    def run(self, schedule: list) -> dict:
        """
        Run N simulations of the full season.
        Returns: {team: [win_totals_across_sims]}
        """
        all_results = defaultdict(list)
        for _ in range(self.N):
            season = self.simulate_season(schedule)
            for team in set(t for game in schedule for t in game):
                all_results[team].append(season.get(team, 0))
        return dict(all_results)

    def playoff_probability(self, results: dict, threshold: int = 42) -> dict:
        """Probability each team reaches a win threshold."""
        probs = {}
        for team, wins in results.items():
            probs[team] = np.mean([w >= threshold for w in wins])
        return dict(sorted(probs.items(), key=lambda x: -x[1]))

    def win_distribution(self, results: dict, team: str) -> tuple:
        """Return mean, std, and full distribution for a team."""
        wins = results[team]
        return np.mean(wins), np.std(wins), wins


# ═══════════════════════════════════════════════════════════════════
#  MODULE 5 — PARLAY RISK ANALYZER
# ═══════════════════════════════════════════════════════════════════

class ParlayAnalyzer:
    """
    Analyze compound probability decay and expected value of parlays.
    P(win all) = p_1 * p_2 * ... * p_n
    """

    @staticmethod
    def parlay_probability(probs: list) -> float:
        """Joint win probability for independent legs."""
        return np.prod(probs)

    @staticmethod
    def parlay_payout(american_odds_list: list, stake: float = 100.0) -> float:
        """Calculate parlay payout from list of American odds."""
        decimal_total = 1.0
        for odds in american_odds_list:
            decimal_total *= OddsConverter.american_to_decimal(odds)
        return stake * decimal_total

    @staticmethod
    def parlay_ev(true_probs: list, american_odds_list: list,
                  stake: float = 100.0) -> float:
        """Expected value of a parlay."""
        p_win = ParlayAnalyzer.parlay_probability(true_probs)
        payout = ParlayAnalyzer.parlay_payout(american_odds_list, stake)
        return p_win * payout - stake

    @staticmethod
    def house_edge_by_legs(n_max: int = 10, single_vig: float = 0.048) -> list:
        """
        Show how house edge compounds with parlay legs.
        single_vig: typical vig per leg (~4.8% for -110/-110 market)
        """
        edges = []
        for n in range(1, n_max + 1):
            fair_prob = 0.5 ** n
            # With vig, implied prob per leg = 0.5 + single_vig/2
            vig_prob_per_leg = 0.5 + single_vig / 2.0
            implied_parlay_prob = vig_prob_per_leg ** n
            house_edge = (implied_parlay_prob - fair_prob) / fair_prob
            edges.append((n, fair_prob, implied_parlay_prob, house_edge))
        return edges


# ═══════════════════════════════════════════════════════════════════
#  MODULE 6 — MARKET EFFICIENCY ANALYZER
# ═══════════════════════════════════════════════════════════════════

class MarketEfficiency:
    """
    Test whether closing lines are well-calibrated:
    games with implied prob ~60% should be won ~60% of the time.
    """

    @staticmethod
    def calibration_test(predictions: list, outcomes: list,
                         n_bins: int = 10) -> dict:
        """
        Bin predictions by probability, compare to actual win rate.
        predictions: list of predicted probabilities
        outcomes: list of 0/1 actual outcomes
        Returns: {bin_center: (predicted_avg, actual_avg, count)}
        """
        preds = np.array(predictions)
        outs = np.array(outcomes)
        bins = np.linspace(0, 1, n_bins + 1)
        results = {}
        for i in range(n_bins):
            mask = (preds >= bins[i]) & (preds < bins[i+1])
            if mask.sum() > 0:
                center = (bins[i] + bins[i+1]) / 2.0
                results[center] = (preds[mask].mean(),
                                   outs[mask].mean(),
                                   int(mask.sum()))
        return results

    @staticmethod
    def brier_score(predictions: list, outcomes: list) -> float:
        """Brier score: mean squared error of probabilistic predictions."""
        preds = np.array(predictions)
        outs = np.array(outcomes)
        return np.mean((preds - outs) ** 2)

    @staticmethod
    def log_loss(predictions: list, outcomes: list,
                 eps: float = 1e-15) -> float:
        """Logarithmic loss (cross-entropy)."""
        preds = np.clip(predictions, eps, 1 - eps)
        preds = np.array(preds)
        outs = np.array(outcomes)
        return -np.mean(outs * np.log(preds) + (1-outs) * np.log(1-preds))


# ═══════════════════════════════════════════════════════════════════
#  MODULE 7 — KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════

class KellyCriterion:
    """
    Optimal bet sizing: f* = (bp - q) / b
    where b = net decimal odds, p = true win prob, q = 1-p
    """

    @staticmethod
    def full_kelly(true_prob: float, american_odds: float) -> float:
        """Full Kelly fraction of bankroll to wager."""
        dec = OddsConverter.american_to_decimal(american_odds)
        b = dec - 1.0  # net odds
        p = true_prob
        q = 1.0 - p
        f = (b * p - q) / b
        return max(f, 0.0)  # never bet negative fraction

    @staticmethod
    def fractional_kelly(true_prob: float, american_odds: float,
                         fraction: float = 0.5) -> float:
        """Fractional Kelly (more conservative)."""
        return fraction * KellyCriterion.full_kelly(true_prob, american_odds)

    @staticmethod
    def bankroll_simulation(initial: float, bets: list,
                            kelly_frac: float = 0.5) -> list:
        """
        Simulate bankroll growth over a series of bets.
        bets: list of (true_prob, american_odds, outcome) tuples
        outcome: 1 = win, 0 = loss
        """
        bankroll = [initial]
        current = initial
        for true_p, odds, outcome in bets:
            f = KellyCriterion.fractional_kelly(true_p, odds, kelly_frac)
            wager = current * f
            dec = OddsConverter.american_to_decimal(odds)
            if outcome == 1:
                current += wager * (dec - 1.0)
            else:
                current -= wager
            bankroll.append(max(current, 0))
            if current <= 0:
                break
        return bankroll


# ═══════════════════════════════════════════════════════════════════
#  MODULE 8 — SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

class DataGenerator:
    """Generate realistic synthetic NBA data for testing."""

    NBA_TEAMS = [
        "Boston Celtics", "Milwaukee Bucks", "Philadelphia 76ers",
        "Cleveland Cavaliers", "New York Knicks", "Brooklyn Nets",
        "Miami Heat", "Atlanta Hawks", "Chicago Bulls",
        "Indiana Pacers", "Toronto Raptors", "Orlando Magic",
        "Detroit Pistons", "Charlotte Hornets", "Washington Wizards",
        "Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder",
        "Dallas Mavericks", "Phoenix Suns", "Los Angeles Lakers",
        "LA Clippers", "Sacramento Kings", "Golden State Warriors",
        "Houston Rockets", "New Orleans Pelicans", "Memphis Grizzlies",
        "San Antonio Spurs", "Utah Jazz", "Portland Trail Blazers"
    ]

    # Approximate true strength tiers
    TEAM_STRENGTHS = {
        "Boston Celtics": 1620, "Oklahoma City Thunder": 1610,
        "Cleveland Cavaliers": 1600, "Denver Nuggets": 1590,
        "Milwaukee Bucks": 1580, "Minnesota Timberwolves": 1570,
        "New York Knicks": 1560, "Dallas Mavericks": 1555,
        "Philadelphia 76ers": 1550, "Phoenix Suns": 1545,
        "Miami Heat": 1530, "Golden State Warriors": 1525,
        "Los Angeles Lakers": 1520, "LA Clippers": 1515,
        "Sacramento Kings": 1510, "Indiana Pacers": 1505,
        "New Orleans Pelicans": 1500, "Atlanta Hawks": 1495,
        "Memphis Grizzlies": 1490, "Chicago Bulls": 1485,
        "Houston Rockets": 1480, "Toronto Raptors": 1475,
        "Brooklyn Nets": 1460, "Orlando Magic": 1455,
        "Utah Jazz": 1440, "Portland Trail Blazers": 1435,
        "San Antonio Spurs": 1425, "Charlotte Hornets": 1420,
        "Detroit Pistons": 1410, "Washington Wizards": 1400,
    }

    @staticmethod
    def generate_schedule(teams: list = None, games_per_matchup: int = 3) -> list:
        """Generate a round-robin schedule with home/away."""
        if teams is None:
            teams = DataGenerator.NBA_TEAMS
        schedule = []
        for i, t1 in enumerate(teams):
            for j, t2 in enumerate(teams):
                if i != j:
                    for g in range(games_per_matchup):
                        if g % 2 == 0:
                            schedule.append((t1, t2))
                        else:
                            schedule.append((t2, t1))
        np.random.shuffle(schedule)
        return schedule

    @staticmethod
    def generate_season_results(elo: EloSystem, schedule: list,
                                noise: float = 0.05) -> list:
        """
        Simulate game results with noise.
        Returns: list of (home, away, home_won, home_score, away_score,
                          market_odds_home, market_odds_away)
        """
        results = []
        for home, away in schedule:
            p_home = elo.win_probability(home, away, home_a=True)
            # Add noise to simulate uncertainty
            p_actual = np.clip(p_home + np.random.normal(0, noise), 0.05, 0.95)
            home_won = 1 if np.random.random() < p_actual else 0

            # Simulate scores
            if home_won:
                home_score = np.random.randint(100, 130)
                away_score = np.random.randint(85, home_score)
            else:
                away_score = np.random.randint(100, 130)
                home_score = np.random.randint(85, away_score)

            # Generate market odds (with vig ~ 4-5%)
            vig = np.random.uniform(0.03, 0.06)
            implied_home = p_home + vig / 2
            implied_away = (1 - p_home) + vig / 2

            # Convert to American odds
            if implied_home >= 0.5:
                odds_home = -round(implied_home / (1 - implied_home) * 100)
            else:
                odds_home = round((1 - implied_home) / implied_home * 100)

            if implied_away >= 0.5:
                odds_away = -round(implied_away / (1 - implied_away) * 100)
            else:
                odds_away = round((1 - implied_away) / implied_away * 100)

            results.append({
                "home": home, "away": away,
                "home_won": home_won,
                "home_score": home_score, "away_score": away_score,
                "market_odds_home": odds_home,
                "market_odds_away": odds_away,
                "true_prob_home": p_home,
                "vig_pct": vig * 100
            })
        return results


# ═══════════════════════════════════════════════════════════════════
#  MODULE 9 — BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════

class Backtester:
    """
    Backtest a betting strategy against historical/simulated data.
    Strategy: bet when model probability exceeds market implied by a threshold.
    """

    def __init__(self, bankroll: float = 10000.0, kelly_fraction: float = 0.25,
                 edge_threshold: float = 0.03):
        self.initial_bankroll = bankroll
        self.kelly_frac = kelly_fraction
        self.edge_threshold = edge_threshold

    def run(self, games: list, elo: EloSystem) -> dict:
        """
        Backtest over game data.
        Returns detailed results.
        """
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        bets_placed = 0
        bets_won = 0
        total_wagered = 0.0
        total_profit = 0.0
        ev_history = []
        bet_details = []

        for game in games:
            home, away = game["home"], game["away"]
            model_prob = elo.win_probability(home, away, home_a=True)
            market_implied = OddsConverter.american_to_implied(
                game["market_odds_home"])

            edge = model_prob - market_implied

            # Check both sides for +EV
            bet_side = None
            bet_odds = None
            bet_prob = None

            if edge > self.edge_threshold:
                bet_side = "home"
                bet_odds = game["market_odds_home"]
                bet_prob = model_prob
            elif -edge > self.edge_threshold:
                bet_side = "away"
                bet_odds = game["market_odds_away"]
                bet_prob = 1.0 - model_prob

            if bet_side and bankroll > 0:
                f = KellyCriterion.fractional_kelly(
                    bet_prob, bet_odds, self.kelly_frac)
                wager = bankroll * f
                if wager < 1.0:
                    # Update Elo
                    mov = abs(game["home_score"] - game["away_score"])
                    elo.update(home, away, game["home_won"],
                               home_a=True, mov=mov)
                    bankroll_history.append(bankroll)
                    continue

                total_wagered += wager
                bets_placed += 1
                dec = OddsConverter.american_to_decimal(bet_odds)
                ev = EVCalculator.expected_value(bet_prob, bet_odds, wager)
                ev_history.append(ev)

                won = (bet_side == "home" and game["home_won"] == 1) or \
                      (bet_side == "away" and game["home_won"] == 0)

                if won:
                    profit = wager * (dec - 1.0)
                    bankroll += profit
                    total_profit += profit
                    bets_won += 1
                else:
                    bankroll -= wager
                    total_profit -= wager

                bet_details.append({
                    "game": f"{away} @ {home}",
                    "side": bet_side,
                    "odds": bet_odds,
                    "model_prob": bet_prob,
                    "edge": abs(edge),
                    "wager": wager,
                    "won": won,
                    "profit": profit if won else -wager,
                    "bankroll": bankroll
                })

            # Update Elo after every game
            mov = abs(game["home_score"] - game["away_score"])
            elo.update(home, away, game["home_won"], home_a=True, mov=mov)
            bankroll_history.append(bankroll)

        return {
            "final_bankroll": bankroll,
            "total_return": (bankroll - self.initial_bankroll) / self.initial_bankroll * 100,
            "bets_placed": bets_placed,
            "bets_won": bets_won,
            "win_rate": bets_won / max(bets_placed, 1),
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": total_profit / max(total_wagered, 1) * 100,
            "avg_ev": np.mean(ev_history) if ev_history else 0,
            "bankroll_history": bankroll_history,
            "bet_details": bet_details,
            "max_drawdown": self._max_drawdown(bankroll_history),
            "sharpe_like": self._sharpe_ratio(bankroll_history),
        }

    @staticmethod
    def _max_drawdown(history):
        peak = history[0]
        max_dd = 0
        for val in history:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100

    @staticmethod
    def _sharpe_ratio(history):
        returns = np.diff(history) / np.array(history[:-1])
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(len(returns))


# ═══════════════════════════════════════════════════════════════════
#  MODULE 10 — VISUALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class Visualizer:
    """Generate all figures for the report."""

    COLORS = {
        "navy": "#1E2A3A", "teal": "#2AA198", "gold": "#D4A843",
        "coral": "#E8634A", "purple": "#6C5CE7", "green": "#27AE60",
        "gray": "#95A5A6", "light_bg": "#F8F9FA"
    }

    @staticmethod
    def setup_style():
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#DDE3EA",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#DDE3EA",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
        })

    @staticmethod
    def plot_parlay_decay(save_path: str):
        """Figure 1: Parlay win probability decay."""
        Visualizer.setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        legs = range(1, 11)
        probs_50 = [0.5**n for n in legs]
        probs_60 = [0.6**n for n in legs]
        probs_65 = [0.65**n for n in legs]

        ax1.semilogy(legs, probs_50, 'o-', color=Visualizer.COLORS["coral"],
                     lw=2.5, markersize=8, label="p=0.50 per leg")
        ax1.semilogy(legs, probs_60, 's-', color=Visualizer.COLORS["teal"],
                     lw=2.5, markersize=8, label="p=0.60 per leg")
        ax1.semilogy(legs, probs_65, '^-', color=Visualizer.COLORS["gold"],
                     lw=2.5, markersize=8, label="p=0.65 per leg")

        ax1.set_xlabel("Number of Parlay Legs")
        ax1.set_ylabel("Win Probability (log scale)")
        ax1.set_title("Parlay Win Probability Decay")
        ax1.legend(framealpha=0.9)
        ax1.set_xticks(range(1, 11))
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # House edge compounding
        edges = ParlayAnalyzer.house_edge_by_legs(10)
        legs_e = [e[0] for e in edges]
        he = [e[3]*100 for e in edges]
        ax2.bar(legs_e, he, color=Visualizer.COLORS["navy"], alpha=0.85, width=0.6)
        ax2.set_xlabel("Number of Parlay Legs")
        ax2.set_ylabel("Effective House Edge (%)")
        ax2.set_title("Compounding House Edge on Parlays")
        ax2.set_xticks(range(1, 11))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_elo_distribution(elo: EloSystem, save_path: str):
        """Figure 2: Elo rating distribution across teams."""
        Visualizer.setup_style()
        fig, ax = plt.subplots(figsize=(12, 7))

        rankings = elo.get_rankings()
        teams = [r[0].replace(" ", "\n", 1) for r in rankings]
        ratings = [r[1] for r in rankings]
        colors_list = [Visualizer.COLORS["teal"] if r > 1550
                       else Visualizer.COLORS["gold"] if r > 1480
                       else Visualizer.COLORS["coral"] for r in ratings]

        bars = ax.barh(range(len(teams)), ratings, color=colors_list, alpha=0.85,
                       height=0.7)
        ax.set_yticks(range(len(teams)))
        ax.set_yticklabels(teams, fontsize=7.5)
        ax.set_xlabel("Elo Rating")
        ax.set_title("NBA Team Elo Ratings (Post-Season Simulation)")
        ax.axvline(x=1500, color=Visualizer.COLORS["gray"], linestyle="--",
                   alpha=0.7, label="League Average (1500)")
        ax.legend()
        ax.invert_yaxis()

        for bar, rating in zip(bars, ratings):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                    f"{rating:.0f}", va="center", fontsize=7.5,
                    color=Visualizer.COLORS["navy"])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_monte_carlo(results: dict, team: str, save_path: str):
        """Figure 3: Monte Carlo win total distribution."""
        Visualizer.setup_style()
        fig, ax = plt.subplots(figsize=(10, 5.5))

        mean, std, wins = MonteCarloSimulator(None).win_distribution(
            results, team)

        ax.hist(wins, bins=range(min(wins), max(wins)+2), density=True,
                alpha=0.7, color=Visualizer.COLORS["teal"], edgecolor="white",
                label=f"Simulated ({len(wins):,} sims)")

        # Overlay normal approximation
        x = np.linspace(min(wins), max(wins), 200)
        normal = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
        ax.plot(x, normal, color=Visualizer.COLORS["coral"], lw=2.5,
                label=f"Normal approx. (μ={mean:.1f}, σ={std:.1f})")

        ax.axvline(mean, color=Visualizer.COLORS["navy"], linestyle="--",
                   lw=1.5, alpha=0.8, label=f"Mean = {mean:.1f} wins")
        ax.axvline(42, color=Visualizer.COLORS["gold"], linestyle=":",
                   lw=1.5, alpha=0.8, label="Playoff threshold (42 wins)")

        ax.set_xlabel("Season Wins")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Monte Carlo Season Simulation: {team}")
        ax.legend(framealpha=0.9, fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_backtest(results: dict, save_path: str):
        """Figure 4: Backtest bankroll trajectory."""
        Visualizer.setup_style()
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # (a) Bankroll trajectory
        ax = axes[0, 0]
        bh = results["bankroll_history"]
        ax.plot(bh, color=Visualizer.COLORS["teal"], lw=1.5, alpha=0.9)
        ax.axhline(bh[0], color=Visualizer.COLORS["gray"], linestyle="--",
                   alpha=0.5, label=f"Initial: ${bh[0]:,.0f}")
        ax.fill_between(range(len(bh)), bh[0], bh, alpha=0.15,
                        color=Visualizer.COLORS["teal"])
        ax.set_title("(a) Bankroll Trajectory")
        ax.set_xlabel("Game Number")
        ax.set_ylabel("Bankroll ($)")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

        # (b) Cumulative P&L
        ax = axes[0, 1]
        details = results["bet_details"]
        if details:
            cum_pnl = np.cumsum([d["profit"] for d in details])
            ax.plot(cum_pnl, color=Visualizer.COLORS["navy"], lw=1.5)
            ax.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                           where=np.array(cum_pnl)>=0,
                           alpha=0.2, color=Visualizer.COLORS["green"])
            ax.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                           where=np.array(cum_pnl)<0,
                           alpha=0.2, color=Visualizer.COLORS["coral"])
            ax.axhline(0, color="black", lw=0.5)
        ax.set_title("(b) Cumulative Profit/Loss")
        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

        # (c) Edge distribution
        ax = axes[1, 0]
        if details:
            edges = [d["edge"]*100 for d in details]
            won_edges = [d["edge"]*100 for d in details if d["won"]]
            lost_edges = [d["edge"]*100 for d in details if not d["won"]]
            ax.hist(won_edges, bins=20, alpha=0.6,
                    color=Visualizer.COLORS["green"], label="Won", edgecolor="white")
            ax.hist(lost_edges, bins=20, alpha=0.6,
                    color=Visualizer.COLORS["coral"], label="Lost", edgecolor="white")
        ax.set_title("(c) Distribution of Edges on Bets Placed")
        ax.set_xlabel("Edge (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

        # (d) Win rate by edge bucket
        ax = axes[1, 1]
        if details:
            edges_arr = np.array([d["edge"] for d in details])
            won_arr = np.array([1 if d["won"] else 0 for d in details])
            buckets = np.percentile(edges_arr, [0, 25, 50, 75, 100])
            bucket_labels = []
            bucket_wr = []
            for i in range(len(buckets)-1):
                mask = (edges_arr >= buckets[i]) & (edges_arr < buckets[i+1] + 0.001)
                if mask.sum() > 0:
                    bucket_labels.append(f"{buckets[i]*100:.1f}-{buckets[i+1]*100:.1f}%")
                    bucket_wr.append(won_arr[mask].mean() * 100)
            ax.bar(range(len(bucket_labels)), bucket_wr,
                   color=Visualizer.COLORS["purple"], alpha=0.8, width=0.6)
            ax.set_xticks(range(len(bucket_labels)))
            ax.set_xticklabels(bucket_labels, fontsize=8)
            ax.axhline(50, color=Visualizer.COLORS["gray"], linestyle="--", alpha=0.5)
        ax.set_title("(d) Win Rate by Edge Quartile")
        ax.set_xlabel("Edge Bucket")
        ax.set_ylabel("Win Rate (%)")

        plt.suptitle("Backtest Results: Elo-Based +EV Betting Strategy",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_calibration(calibration: dict, save_path: str):
        """Figure 5: Model calibration plot."""
        Visualizer.setup_style()
        fig, ax = plt.subplots(figsize=(7, 7))

        centers = sorted(calibration.keys())
        predicted = [calibration[c][0] for c in centers]
        actual = [calibration[c][1] for c in centers]
        counts = [calibration[c][2] for c in centers]

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")
        scatter = ax.scatter(predicted, actual, s=[c*3 for c in counts],
                            c=Visualizer.COLORS["teal"], alpha=0.7,
                            edgecolor="white", lw=1.5, zorder=5)
        ax.plot(predicted, actual, color=Visualizer.COLORS["teal"],
                alpha=0.5, lw=1)

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Win Rate")
        ax.set_title("Elo Model Calibration")
        ax.set_xlim(0.3, 0.8)
        ax.set_ylim(0.3, 0.8)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_kelly_simulation(save_path: str):
        """Figure 6: Kelly criterion bankroll paths comparison."""
        Visualizer.setup_style()
        fig, ax = plt.subplots(figsize=(10, 5.5))

        np.random.seed(42)
        n_bets = 500
        true_prob = 0.55
        odds = 100  # even money

        for frac, color, label in [
            (1.0, Visualizer.COLORS["coral"], "Full Kelly (f=1.0)"),
            (0.5, Visualizer.COLORS["teal"], "Half Kelly (f=0.5)"),
            (0.25, Visualizer.COLORS["gold"], "Quarter Kelly (f=0.25)"),
            (0.1, Visualizer.COLORS["purple"], "Tenth Kelly (f=0.1)")
        ]:
            outcomes = (np.random.random(n_bets) < true_prob).astype(int)
            bets = [(true_prob, odds, o) for o in outcomes]
            path = KellyCriterion.bankroll_simulation(10000, bets, frac)
            ax.plot(path, color=color, lw=1.8, alpha=0.85, label=label)

        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Bankroll ($)")
        ax.set_title("Kelly Criterion: Bankroll Paths by Fraction")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_vig_analysis(games: list, save_path: str):
        """Figure 7: Distribution of vig across games."""
        Visualizer.setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        vigs = [g["vig_pct"] for g in games]
        ax1.hist(vigs, bins=30, color=Visualizer.COLORS["navy"], alpha=0.8,
                 edgecolor="white")
        ax1.axvline(np.mean(vigs), color=Visualizer.COLORS["coral"],
                    linestyle="--", lw=2, label=f"Mean = {np.mean(vigs):.2f}%")
        ax1.set_xlabel("Vigorish (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Sportsbook Vig")
        ax1.legend()

        # EV distribution for flat bettors
        evs = []
        for g in games:
            imp = OddsConverter.american_to_implied(g["market_odds_home"])
            ev = EVCalculator.expected_value(g["true_prob_home"],
                                             g["market_odds_home"])
            evs.append(ev)
        ax2.hist(evs, bins=40, color=Visualizer.COLORS["teal"], alpha=0.8,
                 edgecolor="white")
        ax2.axvline(0, color="black", lw=1)
        ax2.axvline(np.mean(evs), color=Visualizer.COLORS["gold"],
                    linestyle="--", lw=2, label=f"Mean EV = ${np.mean(evs):.2f}")
        ax2.set_xlabel("Expected Value ($) per $100 bet")
        ax2.set_ylabel("Frequency")
        ax2.set_title("EV Distribution: Model vs Market")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION — RUN ALL ANALYSES
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  QUANTITATIVE SPORTS BETTING ANALYSIS SYSTEM")
    print("  Running full analysis pipeline...")
    print("=" * 70)

    np.random.seed(2026)

    # ── 1. Initialize Elo System with team strengths ──
    print("\n[1/8] Initializing Elo rating system...")
    elo = EloSystem(k_factor=20, home_advantage=100, initial_rating=1500)
    for team, rating in DataGenerator.TEAM_STRENGTHS.items():
        elo.ratings[team] = rating
    print(f"  → Loaded {len(elo.ratings)} NBA teams")
    print(f"  → Top 3: {elo.get_rankings()[:3]}")

    # ── 2. Generate synthetic season data ──
    print("\n[2/8] Generating synthetic season schedule & results...")
    schedule = DataGenerator.generate_schedule(
        teams=DataGenerator.NBA_TEAMS[:16],  # Use 16 teams for speed
        games_per_matchup=3
    )
    print(f"  → Schedule: {len(schedule)} games")

    # Create a fresh Elo for simulation
    elo_sim = EloSystem(k_factor=20, home_advantage=100)
    for team, rating in DataGenerator.TEAM_STRENGTHS.items():
        elo_sim.ratings[team] = rating

    games = DataGenerator.generate_season_results(elo_sim, schedule)
    print(f"  → Generated {len(games)} game results with market odds")

    # ── 3. Parlay analysis ──
    print("\n[3/8] Running parlay risk analysis...")
    edges = ParlayAnalyzer.house_edge_by_legs(10)
    for n, fp, ip, he in edges[:5]:
        print(f"  → {n}-leg: Fair P={fp:.4f}, Implied P={ip:.4f}, "
              f"House Edge={he*100:.1f}%")
    Visualizer.plot_parlay_decay(f"{OUT}/fig1_parlay_decay.png")
    print("  → Figure 1 saved: Parlay probability decay")

    # ── 4. Elo ratings after season ──
    print("\n[4/8] Processing season through Elo system...")
    elo_bt = EloSystem(k_factor=20, home_advantage=100)
    for team, rating in DataGenerator.TEAM_STRENGTHS.items():
        elo_bt.ratings[team] = rating

    for g in games:
        mov = abs(g["home_score"] - g["away_score"])
        elo_bt.update(g["home"], g["away"], g["home_won"], home_a=True, mov=mov)

    Visualizer.plot_elo_distribution(elo_bt, f"{OUT}/fig2_elo_ratings.png")
    print("  → Figure 2 saved: Elo rating distribution")
    rankings = elo_bt.get_rankings()
    print(f"  → Post-season #1: {rankings[0][0]} ({rankings[0][1]:.0f})")

    # ── 5. Monte Carlo simulation ──
    print("\n[5/8] Running Monte Carlo season simulation (10,000 sims)...")
    mc = MonteCarloSimulator(elo_bt, n_simulations=10000)
    mc_schedule = DataGenerator.generate_schedule(
        teams=DataGenerator.NBA_TEAMS[:16], games_per_matchup=5)
    mc_results = mc.run(mc_schedule)

    top_team = rankings[0][0]
    playoff_probs = mc.playoff_probability(mc_results, threshold=42)
    print(f"  → {top_team} playoff probability: {playoff_probs.get(top_team, 0)*100:.1f}%")

    Visualizer.plot_monte_carlo(mc_results, top_team,
                                 f"{OUT}/fig3_monte_carlo.png")
    print("  → Figure 3 saved: Monte Carlo win distribution")

    # ── 6. Backtest +EV strategy ──
    print("\n[6/8] Backtesting Elo-based +EV strategy...")
    elo_backtest = EloSystem(k_factor=20, home_advantage=100)
    for team, rating in DataGenerator.TEAM_STRENGTHS.items():
        elo_backtest.ratings[team] = rating

    bt = Backtester(bankroll=10000, kelly_fraction=0.25, edge_threshold=0.03)
    bt_results = bt.run(games, elo_backtest)

    print(f"  → Bets placed: {bt_results['bets_placed']}")
    print(f"  → Win rate: {bt_results['win_rate']*100:.1f}%")
    print(f"  → ROI: {bt_results['roi']:.2f}%")
    print(f"  → Final bankroll: ${bt_results['final_bankroll']:,.2f}")
    print(f"  → Max drawdown: {bt_results['max_drawdown']:.1f}%")
    print(f"  → Total return: {bt_results['total_return']:.1f}%")

    Visualizer.plot_backtest(bt_results, f"{OUT}/fig4_backtest.png")
    print("  → Figure 4 saved: Backtest results (4 panels)")

    # ── 7. Model calibration ──
    print("\n[7/8] Running model calibration analysis...")
    predictions = [g["true_prob_home"] for g in games]
    outcomes = [g["home_won"] for g in games]
    cal = MarketEfficiency.calibration_test(predictions, outcomes)
    brier = MarketEfficiency.brier_score(predictions, outcomes)
    ll = MarketEfficiency.log_loss(predictions, outcomes)
    print(f"  → Brier score: {brier:.4f}")
    print(f"  → Log loss: {ll:.4f}")

    Visualizer.plot_calibration(cal, f"{OUT}/fig5_calibration.png")
    print("  → Figure 5 saved: Model calibration")

    # ── 8. Kelly & Vig analysis ──
    print("\n[8/8] Kelly criterion simulation & vig analysis...")
    Visualizer.plot_kelly_simulation(f"{OUT}/fig6_kelly.png")
    print("  → Figure 6 saved: Kelly criterion paths")
    Visualizer.plot_vig_analysis(games, f"{OUT}/fig7_vig.png")
    print("  → Figure 7 saved: Vig distribution")

    # ── Summary Statistics ──
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE — SUMMARY")
    print("=" * 70)

    summary = {
        "games_analyzed": len(games),
        "teams": len(set(g["home"] for g in games)),
        "avg_vig": np.mean([g["vig_pct"] for g in games]),
        "elo_top_team": rankings[0][0],
        "elo_top_rating": rankings[0][1],
        "mc_simulations": 10000,
        "mc_playoff_prob_top": playoff_probs.get(top_team, 0),
        "backtest_bets": bt_results["bets_placed"],
        "backtest_win_rate": bt_results["win_rate"],
        "backtest_roi": bt_results["roi"],
        "backtest_total_return": bt_results["total_return"],
        "backtest_max_drawdown": bt_results["max_drawdown"],
        "backtest_sharpe": bt_results["sharpe_like"],
        "brier_score": brier,
        "log_loss": ll,
    }

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nAll outputs saved to: {OUT}/")
    print("=" * 70)

    return summary, bt_results, games, elo_bt, mc_results, cal


if __name__ == "__main__":
    summary, bt_results, games, elo, mc_results, calibration = main()
