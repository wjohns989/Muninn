"""
Elo rating system for memory retrieval governance.
"""

from typing import Dict, Any

INITIAL_ELO = 1200.0

def calculate_elo_update(
    current_rating: float,
    outcome: float,
    baseline_rating: float = INITIAL_ELO,
    k_factor: float = 32.0,
) -> float:
    """
    Calculate the new Elo rating for a memory based on retrieval outcome.

    Args:
        current_rating: The current Elo rating of the memory.
        outcome: The feedback outcome (0.0 for rejected/unhelpful, 1.0 for accepted/helpful).
        baseline_rating: The theoretical opponent rating (the 'average' memory).
        k_factor: The maximum possible adjustment per update.

    Returns:
        The new Elo rating.
    """
    # Expected score based on the difference between this memory's rating and the baseline
    expected_score = 1.0 / (1.0 + 10 ** ((baseline_rating - current_rating) / 400.0))

    # Calculate new rating
    new_rating = current_rating + k_factor * (outcome - expected_score)

    # Ensure rating doesn't fall below a hard floor (e.g., 100) or skyrocket infinitely
    return max(100.0, min(new_rating, 3000.0))

def elo_to_half_life_multiplier(elo_rating: float, baseline: float = INITIAL_ELO) -> float:
    """
    Convert an Elo rating to a multiplier for the memory's half-life.
    Ratings above baseline increase half-life (slower decay).
    Ratings below baseline decrease half-life (faster decay).
    """
    # Simple exponential mapping: 
    # e.g., every 400 points doubles or halves the half-life.
    # 1600 -> 2.0x, 800 -> 0.5x
    return 2.0 ** ((elo_rating - baseline) / 400.0)

