"""
.. topic:: Summary

    A set of simple random functionalities

    :Code: mature
    :Documentation: mature
    :Author: Thomas Cokelaer <Thomas.Cokelaer@sophia.inria.fr>
    :Revision: $Id$
"""

import random as std_random


def random(*args):
    """returns a list of random values uniformly distributed

    1 or 2 arguments required.

    If only 1 argument is provided, a value between zero and the argument is returned
    using a uniform distribution. If 2 arguments are provided, the returned value will be between
    the two arguments.

    >>> x = random(1.5)
    >>> x = random(1., 2.)
    """

    if len(args) == 1:
        scale = args[0]
        if isinstance(scale, int):
            assert scale >= 0
            return std_random.randint(0, scale - 1)
        elif isinstance(scale, float):
            return scale * std_random.random()
        else:
            raise ValueError("put an error message if we enter here")
    elif len(args) == 2:
        m = args[0]
        M = args[1]
        assert m < M
        return std_random.uniform(m, M)
    else:
        raise ValueError("1 or 2 arguments expected")


def boolean_event(probability):
    """Return True if the random value is less than the  given probability.

    :param probability: a probability in [0,1]
    :rtype: boolean

    ::

        a = boolean_event(0.5)
    """

    assert probability >= 0.0
    assert probability <= 1.0

    return std_random.random() < probability
