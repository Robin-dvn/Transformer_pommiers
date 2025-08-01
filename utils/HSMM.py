
"""
HSMM Utilities

This module provides tools for working with Hidden Semi-Markov Models (HSMM),
including duration matrix generation, model initialization from TOML files,
sequence generation, and the forward algorithm for likelihood computation.

Functions:
    - generate_duration_matrix: Generate duration distribution matrix from TOML.
    - HSMM: HSMM class for sequence generation and analysis.
"""

import toml
import numpy as np
from scipy.stats import nbinom, poisson, binom
from tqdm import tqdm

def generate_duration_matrix(toml_file, prob_cutoff=1e-20):
    """
    Generate a duration distribution matrix from a TOML file.

    Args:
        toml_file: Path to the TOML file.
        prob_cutoff: Truncation threshold for infinite distributions.

    Returns:
        D: Duration distribution matrix.
    """
    # Load the TOML file
    data = toml.load(toml_file)
    
    occupancy_distributions = data['occupancy_distributions']
    Tmax = 0  # Determine the required maximum duration
    
    # Compute the maximum duration based on the distributions
    for dist in occupancy_distributions:
        d_min, d_max = dist['bounds'][0], dist['bounds'][1]
        if d_max == float("inf"):  # Probabilistic truncation
            param = dist['parameter']
            prob = dist.get('probability', None)  # Probability is optional
            
            if dist['distribution'] == 'NEGATIVE_BINOMIAL':
                assert prob != None, "Probability must be specified for a negative binomial distribution"
                d_max_auto = 1
                while nbinom.sf(d_max_auto, param, prob) > prob_cutoff:
                    # Draw the probability from the negative binomial distribution until it is below the probability threshold
                    d_max_auto += 1
                Tmax = max(Tmax, d_max_auto)
            
            elif dist['distribution'] == 'POISSON':
                d_max_auto = 1
                while poisson.sf(d_max_auto, param) > prob_cutoff:
                    d_max_auto += 1
                Tmax = max(Tmax, d_max_auto)
        else:
            Tmax = max(Tmax, int(d_max))
    
    # Initialize the distribution matrix
    N = len(occupancy_distributions)
    D = np.zeros((N+1, Tmax)) # +1 for the absorbing state distribution
    
    # Fill the matrix with the distributions
    for j, dist in enumerate(occupancy_distributions):
        d_min, d_max = int(dist['bounds'][0]), dist['bounds'][1]
        distribution_type = dist["distribution"]

        if dist['distribution'] == 'NEGATIVE_BINOMIAL':
            param = dist['parameter']
            p = dist.get('probability', 0.5)
            durations = np.arange(0, Tmax )
            probs = nbinom.pmf(durations, param, p)
        elif distribution_type == "BINOMIAL":
            durations = np.arange(0, Tmax )
            p = dist.get("probability", 1.)
            bounds = dist.get("bounds", False)
            inf = bounds[0]
            sup = bounds[1]
            probs = binom.pmf(durations, sup-inf, p) # handle distribution bounds (sup - inf) = number of draws
        elif dist['distribution'] == 'POISSON':
            param = dist['parameter']
            durations = np.arange(0, Tmax )
            probs = poisson.pmf(durations, param)
        
        else:
            raise ValueError(f"Unsupported distribution: {dist['distribution']}")
        
    
        # Fill the matrix
        D[j, :len(probs)] = probs

        # Add a row for the absorbing state
        D[6] = np.zeros(Tmax)  # Initialize to 0
        D[6][-1] = 1.0  # All probability on Tmax

    
    return D



class HSMM:
    def __init__(self, toml_file):
        """
        Initialize an HSMM from a TOML file.
        """
        self.data = toml.load(toml_file)
        self.initial_probabilities = np.array(self.data['initial_probabilities'])
        self.transition_probabilities = np.array(self.data['transition_probabilities'])
        self.observation_distributions = np.array(self.data['observation_distributions'])

        # Normalize probabilities to sum to 1
        self.observation_distributions /= self.observation_distributions.sum(axis=1, keepdims=True)
        self.transition_probabilities /= self.transition_probabilities.sum(axis=1, keepdims=True)
        self.initial_probabilities /= self.initial_probabilities.sum(axis=0, keepdims=True)

        # Generate the duration distribution matrix
        self.duration_matrix = generate_duration_matrix(toml_file)
    
    def get_initial_probabilities(self):
        """
        Return the initial probabilities of the HSMM model.

        Returns:
            initial_probabilities: Array of initial probabilities.
        """
        return self.initial_probabilities
    
    def get_transition_matrix(self):
        """
        Return the transition matrix of the HSMM model.

        Returns:
            transition_probabilities: Transition probability matrix.
        """
        return self.transition_probabilities
    
    def get_observation_matrix(self):
        """
        Return the observation distribution matrix of the HSMM model.

        Returns:
            observation_distributions: Observation distribution matrix.
        """
        return self.observation_distributions
    
    def get_duration_matrix(self):
        """
        Return the duration distribution matrix of the HSMM model.

        Returns:
            duration_matrix: Duration distribution matrix.
        """
        return self.duration_matrix
    
    def display_parameters(self):
        """
        Display the parameters of the HSMM model, including initial probabilities,
        transition probabilities, observation distributions, and the duration matrix.
        """
        print("Initial Probabilities:")
        print(self.initial_probabilities)
        print("\nTransition Probabilities:")
        print(self.transition_probabilities)
        print("\nObservation Distributions:")
        print(self.observation_distributions)
        print("\nDuration Matrix:")
        print(self.duration_matrix)

    def generate_sequence(self, nb_zones=100):
        """
        Generate a sequence of states and observations for a given number of zones.

        Args:
            nb_zones: Number of zones to generate (default 100).

        Returns:
            sequence_states: List of generated states.
            sequence_observations: List of generated observations.
        """
        sequence_states = []
        sequence_observations = []

        # Initialization
        current_state = np.random.choice(len(self.initial_probabilities), p=self.initial_probabilities)

        for _ in range(nb_zones):
            sequence_states.append(current_state)
            # Generate the duration for this state
            duration_probs = self.duration_matrix[current_state]
            duration_probs /= duration_probs.sum()  # Normalize to sum to 1
            lower_bound = int(self.data['occupancy_distributions'][current_state]['bounds'][0])  # add distribution offset
            duration = np.random.choice(len(duration_probs), p=duration_probs) + lower_bound

            # Generate observations during the duration
            for _ in range(duration):
                obs_probs = self.observation_distributions[current_state]
                observation = np.random.choice(len(obs_probs), p=obs_probs)
                sequence_observations.append(observation)

            # Transition to a new state
            new_state = np.random.choice(len(self.transition_probabilities), p=self.transition_probabilities[current_state])
            if new_state == 6:  # stop generation at the absorbing state
                break
            current_state = new_state

        return sequence_states, sequence_observations

    def generate_bounded_sequence(self, l_bound, u_bound):
        """
        Generate a sequence of states and observations with length between specified bounds.

        Args:
            l_bound: Lower bound for sequence length.
            u_bound: Upper bound for sequence length.

        Returns:
            sequence_states: List of generated states.
            sequence_observations: List of generated observations.
        """
        sequence_states = None
        sequence_observations = None
        length = u_bound + 1  # initialize sequence length to enter the while loop
        count = 0  # counter to avoid infinite loops

        # generate a sequence and repeat if the length is not within bounds
        while length > u_bound or length < l_bound:
            sequence_states, sequence_observations = self.generate_sequence()
            count += 1
            length = len(sequence_observations)
            if count == 1000:
                print("Unable to generate a sequence within the requested bounds (too many iterations)")
                break

        sequence_observations = [''.join(str(d)) for d in sequence_observations]

        return sequence_states, sequence_observations
    
    def forward_algorithm(self, observations):
        """
        Apply the Forward algorithm in log-space for an HSMM, summing over all possible durations.
        This implementation computes the probability that the observation sequence is generated by the model.

        Args:
            observations: List of observations for which to compute the probability.

        Returns:
            probability: Probability that the observation sequence is generated by the model.
        """
        T = len(observations)
        N = len(self.initial_probabilities)
        D_max = self.duration_matrix.shape[1]  # Durée max modélisée
        eps = 1e-10  # Pour éviter log(0)
        
        # Log transform
        log_initial = np.log(self.initial_probabilities + eps)
        log_transition = np.log(self.transition_probabilities + eps)
        log_duration = np.log(self.duration_matrix + eps)
        log_observation = np.log(self.observation_distributions + eps)
        
        # Precompute log-probabilities of observation for each state and each time step
        observations = np.array(observations)
        emission = log_observation[:, observations]  # Shape (N, T)

        # Accumulate these log-probabilities to facilitate calculation over segments
        cum_emission = np.cumsum(emission, axis=1)  # Shape (N, T)
        
        # log_alpha[t, j] will contain the log probability that the sequence up to t ends with a segment in state j
        log_alpha = np.full((T, N), -np.inf)
        
        # For each time t, consider all segments ending at t for each state j
        for t in range(T):
            for j in range(N):
                somme_d = -np.inf
                if j != N-1:
                    lower_bound = int(self.data['occupancy_distributions'][j]['bounds'][0])
                else:
                    lower_bound = 0
                d_max_possible = min(t + 1, lower_bound + D_max - 1)
                for d in range(lower_bound, d_max_possible + 1):
                    start = t - d + 1
                    # Compute the contribution of observations over the segment [start, t]
                    dur_idx = d - lower_bound
                    if start == 0:
                        log_emis = cum_emission[j, t]
                        # No previous segment, use initialization
                        candidate = log_initial[j] + log_duration[j, dur_idx] + log_emis
                    else:
                        log_emis = cum_emission[j, t] - cum_emission[j, start - 1]
                        # Sum over transitions from all possible states at the end of the previous segment
                        candidate = (np.logaddexp.reduce(log_alpha[start - 1, :] + log_transition[:, j])
                                     + log_duration[j, dur_idx] + log_emis)
                    somme_d = np.logaddexp(somme_d, candidate)
                log_alpha[t, j] = somme_d
        
        log_prob = np.logaddexp.reduce(log_alpha[T - 1, :])
        log_prob = log_prob / T  # Normalize by sequence length
        return np.exp(log_prob)


# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd

    # Initialize the HSMM model from a TOML file
    hsmm_model = HSMM("data/markov/fuji_long_year_1.toml")

    # Example usage of display_parameters
    print("Display HSMM model parameters:")
    hsmm_model.display_parameters()

    # Example of generating a bounded sequence
    print("\nGenerating a bounded sequence of states and observations:")
    bounded_states, bounded_observations = hsmm_model.generate_bounded_sequence(5, 15)
    print("Generated Bounded Sequence of States:", bounded_states)
    print("Generated Bounded Sequence of Observations:", bounded_observations)
    sequence_observations = [int(d) for d in bounded_observations]
    print(sequence_observations)
    # Example usage of the Forward algorithm
    print("\nComputing the probability of an observation sequence with the Forward algorithm:")
    prob_O = hsmm_model.forward_algorithm(sequence_observations)
    print("Normalized Probability of Observed Sequence:", prob_O)

    # Example of sequence analysis (probability computation) from a CSV file
    analyse = True
    if analyse:
        print("\nSequence analysis from a CSV file:")
        df = pd.read_csv("dataset/markov_python_generated_dataset10000.csv")
        df = df[(df["Observation"] == "LARGE") & (df["Year"] == "Y1")]
        # df = df.head(1000)

        pylist = []
        for seq_str in df['Sequence']:
            seq = [int(char) for char in seq_str]  # Convert each character to an integer and wrap in a list
            pylist.append(seq)

        probs = []
        for i in tqdm(range(len(pylist))):
            prob = hsmm_model.forward_algorithm(pylist[i])
            probs.append(np.log(prob))

        dfprobs = pd.DataFrame(probs, columns=["Probs"])
        dfprobs.to_csv("dataset/probs_dataset_sequence_analysis_perso_corrige_a_garder.csv", index=False)
