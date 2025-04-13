from dataclasses import dataclass


@dataclass
class SchedulerParams:
    """
    :param tabu_length: Number of iterations during which a team cannot be selected.
    :param perturbation_length: Check perturbation need every this many iterations.
    :param n_iterations: Number of tabu phase iterations.
    :param m: Minimum number of time slots between 2 games with same pair of teams.
        --> e.g., one game at slot t and the other game at slot t + m is allowed
            but at slot t + m - 1 is disallowed
    :param p: Cost from dummy supply node q to non-dummy demand node.
    :param r_max: Minimum required time slots for 2 games of same team.
        --> e.g., a single team can play a game at slot t and one as from
            slot t + r_max - 1 (as 'r_max' slots range from t to t + r_max - 1)
    :param penalties: Dictionary as {n_days: penalty} where n_days = rest days + 1.
        --> e.g., respective penalty is assigned if already 1 game
            between slot t - n_days and t + n_days excl. t
            Example input: {1: 10, 2: 3, 3: 1}
    :param alpha: Probability of picking perturbation operator 1.
    :param beta: Probability of removing a game in operator 1.
    """

    tabu_length: int = 4
    perturbation_length: int = 50
    n_iterations: int = 1000
    m: int = 14
    p: int = 1000
    r_max: int = 4
    penalties: dict[int, int] | None = None
    alpha: float = 0.5
    beta: float = 0.01
