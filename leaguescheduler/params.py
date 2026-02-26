from dataclasses import dataclass, field

DEFAULT_COST = 1000
DEFAULT_R_MAX = 4
DEFAULT_COST_REST_DAYS = 500

DEFAULT_PENALTIES = {}
DEFAULT_PENALTIES.update({k + 1: 5 for k in range(DEFAULT_R_MAX - 2, 7)})  # <1 week
DEFAULT_PENALTIES.update({k + 1: 0 for k in range(7, 14)})  # <2 weeks
DEFAULT_PENALTIES.update({k + 1: 3 for k in range(14, 21)})  # 2-3 weeks
DEFAULT_PENALTIES.update({k + 1: 10 for k in range(21, 42)})  # 3-6 weeks
# DEFAULT_PENALTIES.update({k + 1: DEFAULT_COST + 10 for k in range(42, 364)})  # >6 weeks


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
    :param cost_excessive_rest_days: Cost for excessive rest days. [not in original paper]
    """

    tabu_length: int = 4
    perturbation_length: int = 50
    n_iterations: int = 10000
    m: int = 7
    p: int = DEFAULT_COST
    r_max: int = DEFAULT_R_MAX
    penalties: dict[int, int] = field(default_factory=lambda: DEFAULT_PENALTIES.copy())
    alpha: float = 0.5
    beta: float = 0.01
    cost_excessive_rest_days: float = DEFAULT_COST_REST_DAYS
