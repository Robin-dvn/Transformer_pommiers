from typing import List, Tuple, Union, TypedDict, Optional
import dataclasses as dc

import numpy as np

MarkovSequence = List[Tuple[Union[int, None], int]]
InitialProbabilities = Tuple[float]
TransitionProbabilities = Tuple[Tuple[float]]
ObservationDistributions = Tuple[Tuple[float]]
OccupancyDistributions = Tuple["OccupancyDistribution"]


class OccupancyDistribution(TypedDict):
    distribution: str
    parameter: Optional[float]
    probability: Optional[float]
    bounds: Tuple[float, float]


@dc.dataclass
class MarkovModel:
    year: int
    length: str
    initial_probabilities: InitialProbabilities
    transition_probabilities: TransitionProbabilities
    occupancy_distributions: OccupancyDistributions
    observation_distributions: ObservationDistributions

    _final_state = 0

    def __post_init__(self):
        self.length = self.length.upper()

        # make sure sum is 1.
        self.initial_probabilities = (
            np.array(self.initial_probabilities)
            / np.array(self.initial_probabilities).sum()
        )
        self.transition_probabilities = np.array(
            self.transition_probabilities
        ) / np.vstack(np.array(self.transition_probabilities).sum(axis=1))
        self.observation_distributions = np.array(
            self.observation_distributions
        ) / np.vstack(np.array(self.observation_distributions).sum(axis=1))

        no_states = self.initial_probabilities.shape[0]
        assert self.transition_probabilities.shape == (no_states, no_states)
        assert self.observation_distributions.shape == (no_states, no_states - 1)
        assert self.transition_probabilities.shape == (no_states, no_states)
        assert (
            len(self.occupancy_distributions) == no_states - 1
        )  # excluding final state

        self._final_state = no_states - 1

    @property
    def final_state(self):
        return self._final_state


class Markov:
    """
    Class to compute hidden semi markov sequences
    """

    _rng: np.random.Generator
    _max_iterations: int
    _medium: Union[MarkovModel, None] = None
    _long: Union[MarkovModel, None] = None
    _minimum_length: int
    _maximum_length: int

    def __init__(
        self,
        generator: Optional[np.random.Generator] = None,
        minimum_length: int = 4,
        maximum_length: int = 70,
        max_iterations: int = 10000,
    ):
        assert maximum_length <= 300
        assert maximum_length > minimum_length
        assert minimum_length > 0

        self._rng = generator or np.random.default_rng(0)
        self._minimum_length = minimum_length
        self._maximum_length = maximum_length
        self._max_iterations = max_iterations

    @property
    def minimum_length(self) -> int:
        return self._minimum_length

    @property
    def maximum_length(self) -> int:
        return self._maximum_length

    def set_models(self, medium: MarkovModel, long: MarkovModel) -> "Markov":
        self._medium = medium
        self._long = long
        return self

    def _markov(
        self, model: MarkovModel, lower_bound: int, upper_bound: int
    ) -> Union[MarkovSequence, None]:
        final_state = model.final_state
        state: int = np.flatnonzero(
            self._rng.multinomial(1, model.initial_probabilities)
        ).item()

        states = [state]
        while True:
            state = np.flatnonzero(
                self._rng.multinomial(1, model.transition_probabilities[state])
            ).item()
            if state == final_state:
                break
            states.append(state)

        occupancies = []
        for state in states:
            occupancy = 0
            distribution = model.occupancy_distributions[state]
            lbound, ubound = distribution["bounds"]
            fn = None
            parameters = tuple()
            if distribution["distribution"] == "NEGATIVE_BINOMIAL":
                fn = self._rng.negative_binomial
                parameter = distribution["parameter"]
                probability = distribution["probability"]
                parameters = (parameter, probability)
            elif distribution["distribution"] == "BINOMIAL":
                fn = self._rng.binomial
                probability = distribution["probability"]
                parameters = (1, probability)
            elif distribution["distribution"] == "POISSON":
                fn = self._rng.poisson
                parameter = distribution["parameter"]
                parameters = (parameter,)
            if fn is None:
                raise ValueError(
                    f"Distribution {distribution['distribution']} not supported"
                )

            while True:
                occupancy = fn(*parameters)
                
                if occupancy >= lbound and occupancy <= ubound:
                    break
            occupancies.append(occupancy)

        # early return if we know we do not satisfy bounds
        length = sum(occupancies)
        if length > upper_bound or length < lower_bound:
            return None

        sequence: MarkovSequence = []
        for state, occupancy in zip(states, occupancies):
            distribution = model.observation_distributions[state]
            for i in range(occupancy):
                observation: int = np.flatnonzero(
                    self._rng.multinomial(1, distribution)
                ).item()
                sequence.append((state, observation))

        return sequence

    def _generate_sequence(
        self, model: MarkovModel, lower_bound: int, upper_bound: int
    ) -> MarkovSequence:
        iterations = 0
        sequence = None

        while sequence is None and iterations < self._max_iterations:
            sequence = self._markov(model, lower_bound, upper_bound)
            iterations += 1

        if sequence is None:
            raise ValueError(
                f"Maximum iteration of {iterations} reached in sequence generation"
            )
        if iterations > 100:
            # print(
            #     f"Warning: Number of iterations in sequence generation very large: {iterations}"
            # )
            pass

        return sequence

    def generate_bounded_medium_sequence(
        self, lower_bound: int, upper_bound: int
    ) -> MarkovSequence:
        """Generate a bounded medium size Hidden-Semi-Markov sequence"""

        if self._medium is None:
            raise ValueError("No medium MarkovModel set.")

        return self._generate_sequence(self._medium, lower_bound, upper_bound)

    def generate_bounded_long_sequence(
        self, lower_bound: int, upper_bound: int
    ) -> MarkovSequence:
        """Generate a bounded medium size Hidden-Semi-Markov sequence"""

        if self._long is None:
            raise ValueError("No long MarkovModel set.")

        return self._generate_sequence(self._long, lower_bound, upper_bound)
