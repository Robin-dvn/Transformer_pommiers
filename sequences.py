from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pathlib
import srandom
from enums import Observation, Zone
from markov import Markov, MarkovSequence

Sequence = List[Tuple[Union[Zone, None], int]]
TerminalFateData = Dict[Tuple[int, Observation], List[float]]

def get_shared_data_path(path: str) -> str:
    return str((pathlib.Path(__file__).parent / "../data" / path).resolve())

def _markov_to_sequence(sequence: MarkovSequence) -> Sequence:
    # sequence are read from end to start: therefor reversed
    return [(Zone(state), observation) for state, observation in reversed(sequence)]


# class Markov:
#     """
#     Class to manage all the markov and hidden semi markov sequences
#     """

#     hsm_medium: Any
#     hsm_long: Any
#     hsm_short: Any
#     maximum_length: int
#     minimum_length: int

#     def __init__(self, maximum_length=70, minimum_length=4, **kwargs):
#         """
#         :param max_sequence_length: the maximum length of markov sequence (default is 100)
#         :param max_length: the maximum length (default is 70)
#         :param min_length: the minimum length (default is 4)

#         :attributes:
#             * hsm_medium
#             * hsm_long
#             * hsm_short
#             * hsm_short
#             * hsm_96_medium
#             * hsm_97_medium
#             * hsm_98_medium
#             * hsm_95_long
#             * hsm_96_long
#             * hsm_97_long
#             * hsm_98_long
#         """
#         assert maximum_length <= 300
#         assert maximum_length > minimum_length
#         assert minimum_length > 0

#         self.max_sequence_length = 100
#         self.maximum_length = maximum_length
#         self.minimum_length = minimum_length
#         self.hsm_medium = None
#         self.hsm_long = None
#         self.hsm_short = None
#         self.hsm_medium1 = None
#         self.hsm_medium2 = None
#         self.hsm_medium3 = None
#         self.hsm_long1 = None
#         self.hsm_long2 = None
#         self.hsm_long3 = None
#         self.hsm_long4 = None


class TerminalFate:
    """
    Class to deal with terminal fate probabilities
    """

    data: TerminalFateData
    codes: Dict[Observation, int]

    def __init__(self, data: Optional[TerminalFateData] = None):
        """Constructor description

        There is one constructor without arguments that simply setup
        the :attr:`codes` attributes that is  a list of possible shoots:
          1. 'large'
          2. 'medium'
          3. 'small'
          4. 'floral'

        """

        if data is not None:
            self.data = data
        else:
            # Modified by Costes on 30-05-2012
            self.data = {
                (1, Observation.LARGE): [0.500, 0.167, 0.000, 0.333],
                (1, Observation.MEDIUM): [0.000, 0.000, 0.000, 1.000],
                (1, Observation.SMALL): [0.100, 0.100, 0.300, 0.500],
                (1, Observation.FLORAL): [0.100, 0.300, 0.600, 0.000],
                (2, Observation.LARGE): [0.246, 0.185, 0.000, 0.569],
                (2, Observation.MEDIUM): [0.016, 0.238, 0.032, 0.714],
                (2, Observation.SMALL): [0.066, 0.067, 0.317, 0.550],
                (2, Observation.FLORAL): [0.317, 0.250, 0.433, 0.000],
                (3, Observation.LARGE): [0.351, 0.106, 0.010, 0.533],
                (3, Observation.MEDIUM): [0.123, 0.148, 0.063, 0.666],
                (3, Observation.SMALL): [0.015, 0.094, 0.453, 0.438],
                (3, Observation.FLORAL): [0.182, 0.249, 0.569, 0.000],
                (4, Observation.LARGE): [0.213, 0.082, 0.000, 0.705],
                (4, Observation.MEDIUM): [0.027, 0.046, 0.016, 0.911],
                (4, Observation.SMALL): [0.000, 0.024, 0.205, 0.771],
                (4, Observation.FLORAL): [0.003, 0.413, 0.584, 0.000],
                (5, Observation.LARGE): [0.100, 0.050, 0.000, 0.850],
                (5, Observation.MEDIUM): [0.000, 0.020, 0.130, 0.850],
                (5, Observation.SMALL): [0.000, 0.000, 0.375, 0.625],
                (5, Observation.FLORAL): [0.008, 0.325, 0.667, 0.000],
                (6, Observation.LARGE): [0.000, 0.100, 0.000, 0.900],
                (6, Observation.MEDIUM): [0.000, 0.050, 0.050, 0.900],
                (6, Observation.SMALL): [0.000, 0.000, 0.350, 0.650],
                (6, Observation.FLORAL): [0.000, 0.200, 0.800, 0.000],
            }

        self.codes = {
            Observation.LARGE: 0,
            Observation.MEDIUM: 1,
            Observation.SMALL: 2,
            Observation.FLORAL: 3,
        }

    def get_data_terminal_fate(self, year_no: int, code: Observation) -> List[float]:
        """Returns the probabilities corresponding to a shoot code and a year

        It uses hardcoded list of probabilities such as::

            (1996, 'medium'): [0.016, 0.238, 0.032, 0.714],


        :param year: if year is less than 1994 or greater than 2000, then 2000 is chosen
        :param code: must be large, medium, small, floral

        :returns: an array containing the probabilities to have a large, medium,
            small or floral sequence.
        """
        # Terminal fate for year 0 and 1 are the same
        if year_no == 0:
            year_no = 1
        elif year_no < 0 or year_no > 6:
            year_no = 6

        if code in self.codes.keys():
            return self.data[(year_no, code)]
        else:
            raise ValueError(
                "code must be in %s. %s provided" % (self.codes.keys(), code)
            )

    def _check_probabilities(self):
        """Check that all arrays sum up to a probability of 1

        for testing usage only.

        :Example:

            >>> d = DataTerminalFate()
            >>> d._check_probabilities()
        """
        for data in self.data.values():
            assert sum(data) == 1


def terminal_fate(
    year_no: int,
    observation: Observation,
    data_terminal_fate: Optional[TerminalFateData] = None,
) -> Observation:
    """This function returns a type of metamer (large, short, ...)


    It uses :class:`~openalea.stocatree.sequences.DataTerminalFate` class.

    :param year_no: is an int starting from 0 for the 1st year of the simulation
    :param observation: is a string. ['large', 'medium','small', 'floral'].
        See :class:`DataTerminalFate` class documentation for details.

    """

    d = TerminalFate(data_terminal_fate)
    data = d.get_data_terminal_fate(year_no, observation)
    index = _non_parametric_distribution(data)

    if index == 1:
        return Observation.LARGE
    if index == 2:
        return Observation.MEDIUM
    if index == 3:
        return Observation.SMALL
    if index == 4:
        return Observation.FLORAL

    raise ValueError(f"Unknown index {index}")


def _non_parametric_distribution(pdf):
    """Returns the index(x) at which the PDF(x) reaches random value in [0,1]

    :param pdf: The PDF function

    :Example:

        >>> res = _non_parametric_distribution([0.25,0.25,0.25,0.25])

    .. note:: The sum of PDF must be 1 (checked by this function)

    """
    assert sum(pdf) == 1
    target = srandom.random(1.0)
    cumulation = 0
    i = 0
    length = len(pdf)
    while i < length and cumulation < target:
        cumulation += pdf[i]
        i += 1
    return i


# def generate_hsm_sequence(hsm, sequence_length=100) -> Sequence:
#     """Generate a Hidden Semi Markov Sequence given an input transition matrix

#     Used by :meth:`~openalea.stocatree.sequences.generate_sequence`

#     :param hsm: A hidden semi markov instance of HiddenSemiMarkov class from VPlants.Sequence_analysis
#     :param sequence_length: a length of sequence set to 100 by default

#     :returns: a sequence

#     """

#     iterator = SemiMarkovIterator(hsm)
#     simulation = iterator.simulation(sequence_length, True)

#     i = 0
#     sequence = []
#     for i in range(0, sequence_length):
#         if simulation[0][i] == Zone.DORMANT_END:  # take last entry of zones instead
#             break
#         sequence.append((simulation[0][i], simulation[1][i]))
#     sequence.reverse()

#     return sequence


# def generate_bounded_hsm_sequence(hsm, lower_bound: int, upper_bound: int) -> Sequence:
#     """Returns a bouded sequence

#     One problem with the Markov chains is that they may produce sequences
#     with an unrealistic number of metamers. when this occurs, the generated
#     sequence is thrown away and a new one is generated.

#     Medium shoots are always limited to five to fifteen metamers.

#     Long shoots ashrinks as the tree ages. They are therefore separated into
#     subcategories: 15-25, 26-40 and over 40 metamers (41 to 70).

#     The probability to select one type of shoot length depends on the year. These
#     probabilities are stored within the hsm data structure

#     :param hsm: a HiddenSemiMarkov instance
#     :param lower_bound: int
#     :param upper_bound: int

#     ::

#         generate_bounded_hsm_sequence(markov.hsm_long,  15, 26);

#     """
#     length = upper_bound + 1  # defines a max length for the sequence
#     count = 0
#     sequence = []

#     while length > upper_bound or length < lower_bound and count < 1000:
#         sequence = generate_hsm_sequence(hsm)
#         length = len(sequence)
#         count += 1

#     if count == 1000:
#         raise ValueError('to be done. max count limit reached in generate_bounded_hsm_sequence')
#     if count > 100:
#         print('Warning, count in generate_bounded_hsm_sequence was large :%d' % count)

#     return sequence


def generate_short_sequence() -> Sequence:
    """Generate a short sequence

    Used by :meth:`~openalea.stocatree.sequences.generate_sequence`

    :rtype: a list of sequences

    :Example:

        >>> seq = generate_short_sequence()

    .. todo:: check the relevance of this function

    """

    return [
        (Zone.DORMANT_START, 0),
        (Zone.DORMANT_START, 0),
        (Zone.DORMANT_START, 0),
        (Zone.DORMANT_START, 0),
    ]


def generate_floral_sequence() -> Sequence:
    """Generate a floral sequence

    Used by :meth:`~openalea.stocatree.sequences.generate_sequence`

    :rtype: a list of sequences
    :Example:

        >>> seq = generate_floral_sequence()

    .. todo:: do not known what this is doing what difference with short_sequence? Seems normal behaviour
    .. todo:: original code uses s(0,12) but was not used
    """

    return [
        (Zone.DORMANT_START, 0),
        (Zone.DORMANT_START, 0),
        (Zone.DORMANT_START, 0),
        (Zone.DORMANT_START, 0),
    ]


def generate_trunk(
    trunk_seq="trunk/sequences.seq", select: Union[List[int], int] = 0
) -> Sequence:
    """Generate a trunk sequence randomly selected within a list of hard-coded trunk sequences

    Used by :meth:`~openalea.stocatree.sequences.generate_sequence` only

    :param list select: the index of the selected trunk in the list of trunk sequence (default is 0).

    :Example:

        >>> deterministic_sequence = generate_trunk(select=1)

    """
    select_ = 0
    if type(select) is list:
        select_ = select[0]
    elif type(select) is int:
        select_ = select
    seqs = np.loadtxt(get_shared_data_path(trunk_seq), int)

    assert 0 <= select_ < len(seqs)

    seq: List[int] = list(seqs[select_])
    sequence: Sequence = []
    for obs in seq:
        if obs == 9:
            break
        sequence.append((None, obs))

    sequence.reverse()

    return sequence


def _generate_random_draw_sequence() -> Sequence:
    """an alternative to the long shoot  model of the 2nd year. Usually not used."""

    max_length = 65
    number = 9

    second_year_branches = [
        [
            0,
            0,
            0,
            0,
            0,
            3,
            2,
            2,
            1,
            1,
            0,
            0,
            0,
            0,
            2,
            0,
            1,
            1,
            4,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            4,
            0,
            4,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            3,
            0,
            0,
            0,
            0,
            0,
            4,
            4,
            4,
            4,
            4,
            3,
            0,
            4,
            4,
            4,
            0,
            4,
            0,
            4,
            4,
            0,
            0,
            4,
            0,
            0,
            0,
            2,
            3,
            0,
            0,
            0,
            3,
            3,
            3,
            3,
            0,
            0,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            3,
            4,
            0,
            1,
            4,
            4,
            1,
            0,
            4,
            0,
            1,
            4,
            4,
            0,
            4,
            4,
            4,
            4,
            4,
            4,
            0,
            4,
            4,
            0,
            0,
            0,
            1,
            0,
            4,
            4,
            4,
            0,
            4,
            0,
            4,
            0,
            0,
            0,
            3,
            0,
            1,
            0,
            0,
            0,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            4,
            1,
            1,
            4,
            3,
            1,
            0,
            0,
            4,
            0,
            0,
            4,
            0,
            0,
            4,
            0,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            0,
            0,
            3,
            0,
            0,
            0,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            2,
            3,
            0,
            0,
            3,
            3,
            0,
            0,
            0,
            1,
            2,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            4,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            4,
            3,
            2,
            4,
            4,
            0,
            2,
            0,
            3,
            0,
            2,
            0,
            0,
            4,
            0,
            4,
            4,
            4,
            4,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            4,
            4,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            2,
            3,
            0,
            0,
            0,
            3,
            2,
            0,
            0,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            1,
            4,
            0,
            4,
            0,
            3,
            0,
            0,
            1,
            3,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            3,
            0,
            0,
            1,
            1,
            0,
            0,
            3,
            4,
            0,
            4,
            0,
            0,
            4,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            3,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            3,
            3,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            1,
            0,
            0,
            4,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
    ]

    select_branch = int(srandom.random(number))
    i = 0
    sequence: Sequence = []
    for i in range(0, max_length):
        if second_year_branches[select_branch][i] == 9:
            break
        sequence.append((None, second_year_branches[select_branch][i]))

    sequence.reverse()

    return sequence


def generate_sequence(
    obs: Observation,
    markov: Markov,
    year_no: int = 0,
    second_year_draws: bool = False,
    trunk_seq: str = "trunk/sequences.seq",
    select_trunk: int = 0,
) -> Sequence:
    """Generation of sequences from Markov chains, based directly on the work
    of Michael Renton

    Bounded sequences:

    One problem with the Markov chains is that they may produce sequences
    with an unrealistic number of metamers. when this occurs, the generated
    sequence is thrown away and a new one is generated.

    Medium shoots are always limited to five to fifteen metamers.

    Long shoots ashrinks as the tree ages. They are therefore separated into
    subcategories: 15-25, 26-40 and over 40 metamers (41 to 70).

    The probability to select one type of shoot length depends on the year. These
    probabilities are stored within the hsm data structure

    if observation is
        * trunk, call :meth:`~openalea.stocatree.sequences.generate_trunk`
        * small, call :meth:`~openalea.stocatree.sequences.generate_short_sequence`
        * floral, call :meth:`~openalea.stocatree.sequences.generate_floral_sequence`
        * medium, call :meth:`~openalea.stocatree.sequences.generate_bounded_hsm_sequence` with bounded values = [5, 15]
        * large, call :meth:`~openalea.stocatree.sequences.generate_bounded_hsm_sequence`



    :param obs: the apex observation in ['trunk', 'small', 'floral', 'medium', 'large']
    :param markov: an instance of :class:`~openalea.stocatree.sequences.Markov`
    :param year: the simulation year, first year is 0
    :param second_year_draws: if True, uses the alternative `_generate_random_sequences`
        function instead of generate_bounded_hsm_sequence. (default is False).
        Can only be set to True if year=1
    :param int select_trunk: selection of trunk sequences within the list. See :func:`generate_trunk`


    :returns: a random sequence
    """
    # This fix the c++ seed at each call with a random number from the uniform law in python
    # were the seed was also fixed. Therefore the successive seeds used in c++ are fixed for
    # a given python seed allowing to reproduce trees.
    # srand(int(random.uniform(0, 1e6)))

    # The "sylleptic"s to the conditions as following were added by Han on 30-04-2012
    if obs == Observation.TRUNK:
        return generate_trunk(trunk_seq=trunk_seq, select=select_trunk)
    elif obs == Observation.SMALL or obs == Observation.SYLLEPTIC_SMALL:
        return generate_short_sequence()
    elif obs == Observation.FLORAL:
        return generate_floral_sequence()
    elif obs == Observation.MEDIUM or obs == Observation.SYLLEPTIC_MEDIUM:
        return _markov_to_sequence(markov.generate_bounded_medium_sequence(5, 15))
        # return generate_bounded_hsm_sequence(markov.hsm_medium, 5, 15)
    elif obs == Observation.LARGE or obs == Observation.SYLLEPTIC_LARGE:
        if second_year_draws and year_no == 1:
            return _generate_random_draw_sequence()
        else:
            res = length_pool(year_no)
            assert res in [1, 2, 3], "Error Bad Length pool category"
            if res == 1:
                return _markov_to_sequence(
                    markov.generate_bounded_long_sequence(15, 26)
                )
                # return generate_bounded_hsm_sequence(markov.hsm_long, 15, 26)
            elif res == 2:
                return _markov_to_sequence(
                    markov.generate_bounded_long_sequence(26, 41)
                )
                # return generate_bounded_hsm_sequence(markov.hsm_long, 26, 41)
            elif res == 3:
                return _markov_to_sequence(
                    markov.generate_bounded_long_sequence(41, markov._maximum_length)
                )
                # return generate_bounded_hsm_sequence(markov.hsm_long, 41, markov.maximum_length)

    raise ValueError(
        "A bad sequence observation (%s) was passed to generate_sequence().\n" % obs
    )


# currently usused
# def generate_pruned_sequence(
#     obs: Observation,
#     react_pos,
#     rank,
#     closest_apex,
#     farthest_apex,
#     sons_nb,
#     markov: Markov,
#     year=0
# ):
#     """The pruned length is assimilated to the distance to the farthest apex

#     obs is the shoot type of the prunned shoot
#     react_pos is the position of the reacting apex from the cuting point [0,2]
#     year is the year pruned shoot creation - start date year

#     Determining the case of pruning reaction for the 3 react-pos
#      A : Succession - Succession - lower Cat
#      B : Reiteration - Succession - lower Cat
#      C : Reiteration - Reiteration - Succession


#     Case of pruning a shoot w/o branching
#     Type of shoot is generated according to the pruned length
#     """

#     if closest_apex == farthest_apex:
#         # ratio of pruned over total length
#         pruned_ratio = 1.0 * farthest_apex / (rank + farthest_apex)
#         if pruned_ratio < 0.25:
#             case = 'A'
#         elif pruned_ratio < 0.75:
#             case = 'B'
#         else:
#             case = 'C'
#         new_obs = shoot_type_react(year, obs, case, react_pos)
#     else:
#         # Case of pruning a shoot with branching. Then depending on the pruned length
#         # and biomass represented by the sons_nb.
#         # ratio of total biomass(sons_nb) over pruned length(farthest_apex)
#         bio_ratio = 1.0 * sons_nb / farthest_apex
#         if bio_ratio < 2:
#             case = 'A'
#         if bio_ratio < 3:
#             case = 'B'
#         else:
#             case = 'C'

#         new_obs = shoot_type_react(year, obs, case, react_pos)

#     hsm_react_long, hsm_react_medium = pruned_hsmc(year, markov)

#     if new_obs in (Observation.TRUNK, Observation.LARGE, Observation.SYLLEPTIC_LARGE):
#         if farthest_apex > 30:
#             return generate_bounded_hsm_sequence(hsm_react_long, 41, markov.maximum_length)
#         elif farthest_apex > 20:
#             return generate_bounded_hsm_sequence(hsm_react_long, 26, 41)
#         elif farthest_apex > 8:
#             return generate_bounded_hsm_sequence(hsm_react_long, 15, 26)
#         else:
#             return generate_bounded_hsm_sequence(hsm_react_medium, 5, 15)

#     elif new_obs in (Observation.MEDIUM, Observation.SYLLEPTIC_MEDIUM):
#         if farthest_apex > 5:
#             return generate_bounded_hsm_sequence(hsm_react_long, 15, 26)
#         else:
#             return generate_bounded_hsm_sequence(hsm_react_medium, 5, 15)

#     elif new_obs in (Observation.SMALL, Observation.SYLLEPTIC_SMALL):
#         return generate_short_sequence()

#     elif new_obs == Observation.FLORAL:
#         return generate_floral_sequence()


def shoot_type_react(
    year, pruned_shoot_type: Observation, pruning_case, react_pos
) -> Observation:
    if pruned_shoot_type == Observation.TRUNK:
        pruned_shoot_type = Observation.LARGE
        reiteration = Observation.LARGE
        succession = Observation.LARGE
        lower_cat = terminal_fate(year, pruned_shoot_type)
    else:
        reiteration = pruned_shoot_type
        succession = terminal_fate(year, pruned_shoot_type)
        lower_cat = terminal_fate(year, terminal_fate(year, pruned_shoot_type))

    if react_pos == 0:
        if pruning_case == "A":
            print(
                "reaction at pos {0} in case {1} of type {2}".format(
                    react_pos, pruning_case, succession
                )
            )
            return succession
        else:
            print(
                "reaction at pos {0} in case {1} of type {2}".format(
                    react_pos, pruning_case, reiteration
                )
            )
            return reiteration
    elif react_pos == 1:
        if pruning_case == "C":
            print(
                "reaction at pos {0} in case {1} of type {2}".format(
                    react_pos, pruning_case, reiteration
                )
            )
            return reiteration
        else:
            print(
                "reaction at pos {0} in case {1} of type {2}".format(
                    react_pos, pruning_case, succession
                )
            )
            return succession
    elif react_pos == 2:
        if pruning_case == "C":
            print(
                "reaction at pos {0} in case {1} of type {2}".format(
                    react_pos, pruning_case, succession
                )
            )
            return succession
        else:
            print(
                "reaction at pos {0} in case {1} of type {2}".format(
                    react_pos, pruning_case, lower_cat
                )
            )
            return lower_cat

    raise ValueError(f"Reaction at {react_pos} not implemented")


def pruned_hsmc(year, markov):
    """
    Return a long and a medium hsmc model from the year before the current one
    """

    if year in [0, 1]:
        hsmc_medium = markov.hsm_medium1
        hsmc_long = markov.hsm_long1
    elif year == 2:
        hsmc_medium = markov.hsm_medium2
        hsmc_long = markov.hsm_long2
    elif year == 3:
        hsmc_medium = markov.hsm_medium3
        hsmc_long = markov.hsm_long3
    else:
        hsmc_medium = markov.hsm_medium4
        hsmc_long = markov.hsm_long4

    return hsmc_long, hsmc_medium


def length_pool(year: int):
    """Returns a random number according to `year`
    If the trunk was cut far enough from the top, generate the longest possible shoot, otherwise,
    depending on the pruned length, i.e. farthest apex. year should come from simulation.year
    """

    pool_1995 = [0.111, 0.222, 0.667]
    pool_1996 = [0.538, 0.346, 0.116]
    pool_1997 = [0.830, 0.170, 0.000]
    pool_1998 = [0.940, 0.060, 0.000]
    pool_1999 = [0.965, 0.035, 0.000]

    if year == 0 or year == 1:
        return _non_parametric_distribution(pool_1995)
    elif year == 2:
        return _non_parametric_distribution(pool_1996)
    elif year == 3:
        return _non_parametric_distribution(pool_1997)
    elif year == 4:
        return _non_parametric_distribution(pool_1998)
    else:
        return _non_parametric_distribution(pool_1999)
