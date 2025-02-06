import toml
import numpy as np
from scipy.stats import nbinom, poisson,binom

def generate_duration_matrix(toml_file, prob_cutoff=1e-6):
    """
    Génère une matrice de distribution de durée à partir d'un fichier TOML.
    
    Paramètres :
    - toml_file : chemin vers le fichier TOML
    - prob_cutoff : seuil de troncature pour les distributions infinies
    
    Retourne :
    - D : Matrice des distributions de durée
    """
    # Charger le fichier TOML
    data = toml.load(toml_file)
    
    occupancy_distributions = data['occupancy_distributions']
    Tmax = 0  # Déterminer la durée maximale requise
    
    # Calculer la durée maximale en fonction des distributions
    for dist in occupancy_distributions:
        d_min, d_max = dist['bounds'][0], dist['bounds'][1]
        if d_max == float("inf"):  # Troncature probabiliste
            param = dist['parameter']
            prob = dist.get('probability', None)  # La probabilité est facultative
            
            if dist['distribution'] == 'NEGATIVE_BINOMIAL':
                p = prob if prob else 0.5  # Probabilité par défaut
                d_max_auto = 1
                while nbinom.sf(d_max_auto, param, p) > prob_cutoff:
                    d_max_auto += 1
                Tmax = max(Tmax, d_max_auto)
            
            elif dist['distribution'] == 'POISSON':
                d_max_auto = 1
                while poisson.sf(d_max_auto, param) > prob_cutoff:
                    d_max_auto += 1
                Tmax = max(Tmax, d_max_auto)
        else:
            Tmax = max(Tmax, int(d_max))
    
    # Initialiser la matrice des distributions
    N = len(occupancy_distributions)
    D = np.zeros((N+1, Tmax))
    
    # Remplir la matrice avec les distributions
    for j, dist in enumerate(occupancy_distributions):
        d_min, d_max = int(dist['bounds'][0]), dist['bounds'][1]
        distribution_type = dist["distribution"]

        if dist['distribution'] == 'NEGATIVE_BINOMIAL':
            param = dist['parameter']
            p = dist.get('probability', 0.5)
            durations = np.arange(1, Tmax + 1)
            probs = nbinom.pmf(durations, param, p)
        elif distribution_type == "BINOMIAL":
            durations = np.arange(1, Tmax + 1)
            p = dist.get("probability", 1.)
            probs = binom.pmf(durations, d_max, p) 
        elif dist['distribution'] == 'POISSON':
            param = dist['parameter']
            durations = np.arange(1, Tmax + 1)
            probs = poisson.pmf(durations, param)
        
        else:
            raise ValueError(f"Distribution non supportée : {dist['distribution']}")
        
        # Appliquer les bornes
        probs[:d_min-1] = 0  # Mettre à zéro les durées < d_min
        if d_max != float("inf"):
            probs[int(d_max):] = 0  # Mettre à zéro les durées > d_max
        
        # Normalisation
        if probs.sum() > 0:
            probs /= probs.sum()

        
        # Remplir la matrice
        D[j, :len(probs)] = probs
        D[6] = np.zeros(Tmax)  # Initialisation à 0
        D[6][-1] = 1.0  # Toute la probabilité sur Tmax

    
    return D


# class HSMM:
#     def __init__(self, toml_file):
#         """
#         Initialise un modèle de Markov semi-caché (HSMM) à partir d'un fichier TOML.
#         """
#         self.data = toml.load(toml_file)
#         self.initial_probabilities = np.array(self.data['initial_probabilities'])
#         self.transition_probabilities = np.array(self.data['transition_probabilities'])
#         self.observation_distributions = np.array(self.data['observation_distributions'])
#         self.observation_distributions /= self.observation_distributions.sum(axis=1, keepdims=True)
#         self.transition_probabilities /= self.transition_probabilities.sum(axis=1, keepdims=True)
#         self.initial_probabilities /= self.initial_probabilities.sum(axis=0, keepdims=True)
#         # Générer la matrice de distribution de durée
#         self.duration_matrix = generate_duration_matrix(toml_file)
    
#     def get_initial_probabilities(self):
#         """Retourne les probabilités initiales."""
#         return self.initial_probabilities
    
#     def get_transition_matrix(self):
#         """Retourne la matrice de transition des états cachés."""
#         return self.transition_probabilities
    
#     def get_observation_matrix(self):
#         """Retourne la matrice des distributions d'observation."""
#         return self.observation_distributions
    
#     def get_duration_matrix(self):
#         """Retourne la matrice des distributions de durée."""
#         return self.duration_matrix
    
#     def display_parameters(self):
#         """Affiche les paramètres du HSMM."""
#         print("Initial Probabilities:")
#         print(self.initial_probabilities)
#         print("\nTransition Probabilities:")
#         print(self.transition_probabilities)
#         print("\nObservation Distributions:")
#         print(self.observation_distributions)
#         print("\nDuration Matrix:")
#         print(self.duration_matrix)
    
#     def generate_sequence(self, min_length, max_length):
#         """Génère une séquence d'états et d'observations avec une longueur entre min_length et max_length."""
#         sequence_length = np.random.randint(min_length, max_length + 1)
#         sequence_states = []
#         sequence_observations = []
        
#         # Initialiser l'état
#         current_state = np.random.choice(len(self.initial_probabilities), p=self.initial_probabilities)
        
#         while len(sequence_states) < sequence_length:
#             sequence_states.append(current_state)
            
#             # Générer la durée de cet état
#             duration_probs = self.duration_matrix[current_state]
#             duration = np.random.choice(len(duration_probs), p=duration_probs) + 1
#                         # Vérifier si l'état absorbant est atteint

#             # Générer les observations pour cette durée
#             for _ in range(duration):
#                 if len(sequence_observations) >= sequence_length:
#                     break
#                 observation_probs = self.observation_distributions[current_state]

#                 observation = np.random.choice(len(observation_probs), p=observation_probs)
#                 sequence_observations.append(observation)
            
#             # Transition vers le nouvel état
#             new_state = np.random.choice(len(self.transition_probabilities), p=self.transition_probabilities[current_state])
#             if new_state   == 6:
#                 break
#             current_state = new_state
        
#         return sequence_states, sequence_observations

#     def forward_algorithm(self, observations):
#         """Applique l'algorithme Forward en log-space pour éviter l'underflow."""
#         T = len(observations)
#         N = len(self.initial_probabilities)
        
#         log_alpha = np.full((N, T), -np.inf)  # Matrice des probabilités en log-space
#         log_initial = np.log(self.initial_probabilities + 1e-10)  # Éviter log(0)
#         log_transition = np.log(self.transition_probabilities + 1e-10)
#         log_observation = np.log(self.observation_distributions + 1e-10)
        
#         # Initialisation
#         for i in range(N):
#             log_alpha[i, 0] = log_initial[i] + log_observation[i, observations[0]]
        
#         # Récurrence
#         for t in range(1, T):
#             for j in range(N):
#                 log_alpha[j, t] = np.logaddexp.reduce(
#                     log_alpha[:, t - 1] + log_transition[:, j]
#                 ) + log_observation[j, observations[t]]
        
#         # Finalisation
#         log_prob_O = np.logaddexp.reduce(log_alpha[:, -1])
#         return np.exp(log_prob_O)  # Convertir en probabilité réelle



class HSMM:
    def __init__(self, toml_file):
        """
        Initialise un HSMM à partir d'un fichier TOML.
        """
        self.data = toml.load(toml_file)
        self.initial_probabilities = np.array(self.data['initial_probabilities'])
        self.transition_probabilities = np.array(self.data['transition_probabilities'])
        self.observation_distributions = np.array(self.data['observation_distributions'])
        self.observation_distributions /= self.observation_distributions.sum(axis=1, keepdims=True)
        self.transition_probabilities /= self.transition_probabilities.sum(axis=1, keepdims=True)
        self.initial_probabilities /= self.initial_probabilities.sum(axis=0, keepdims=True)
        # Génère la matrice de distribution de durée (supposée définie ailleurs)
        self.duration_matrix = generate_duration_matrix(toml_file)
    
    def get_initial_probabilities(self):
        return self.initial_probabilities
    
    def get_transition_matrix(self):
        return self.transition_probabilities
    
    def get_observation_matrix(self):
        return self.observation_distributions
    
    def get_duration_matrix(self):
        return self.duration_matrix
    
    def display_parameters(self):
        print("Initial Probabilities:")
        print(self.initial_probabilities)
        print("\nTransition Probabilities:")
        print(self.transition_probabilities)
        print("\nObservation Distributions:")
        print(self.observation_distributions)
        print("\nDuration Matrix:")
        print(self.duration_matrix)
    
    def generate_sequence(self, min_length, max_length):
        sequence_length = np.random.randint(min_length, max_length + 1)
        sequence_states = []
        sequence_observations = []
        
        # Initialisation
        current_state = np.random.choice(len(self.initial_probabilities), p=self.initial_probabilities)
        
        while len(sequence_states) < sequence_length:
            sequence_states.append(current_state)
            # Générer la durée pour cet état
            duration_probs = self.duration_matrix[current_state]
            duration = np.random.choice(len(duration_probs), p=duration_probs) + 1
            # Générer les observations pendant la durée
            for _ in range(duration):
                if len(sequence_observations) >= sequence_length:
                    break
                obs_probs = self.observation_distributions[current_state]
                observation = np.random.choice(len(obs_probs), p=obs_probs)
                sequence_observations.append(observation)
            # Transition vers un nouvel état
            new_state = np.random.choice(len(self.transition_probabilities), p=self.transition_probabilities[current_state])
            if new_state == 6:  # par exemple, état absorbant
                break
            current_state = new_state
        
        return sequence_states, sequence_observations

    def forward_algorithm(self, observations):
        """
        Applique l'algorithme Forward en log-espace pour un HSMM, en sommant sur toutes les durées possibles.
        Cette implémentation calcule la probabilité que la séquence d'observations soit générée par le modèle.
        """
        T = len(observations)
        N = len(self.initial_probabilities)
        D_max = self.duration_matrix.shape[1]  # Durée max modélisée
        eps = 1e-10  # Pour éviter log(0)
        
        # Passage en log
        log_initial = np.log(self.initial_probabilities + eps)
        log_transition = np.log(self.transition_probabilities + eps)
        log_duration = np.log(self.duration_matrix + eps)
        log_observation = np.log(self.observation_distributions + eps)
        
        # Pré-calculer les log-probabilités d'observation pour chaque état et chaque instant
        # emission[j, t] = log b_j(observations[t])
        observations = np.array(observations)
        emission = log_observation[:, observations]  # Shape (N, T)
        # Cumuler ces log-probabilités pour faciliter le calcul sur des segments
        cum_emission = np.cumsum(emission, axis=1)  # Shape (N, T)
        
        # log_alpha[t, j] contiendra le log de la probabilité que la séquence jusqu'à t se termine par un segment d'état j
        log_alpha = np.full((T, N), -np.inf)
        
        # Pour chaque instant t, on considère tous les segments se terminant en t pour chaque état j
        for t in range(T):
            for j in range(N):
                somme_d = -np.inf
                max_d = min(D_max, t + 1)  # Un segment ne peut être plus long que t+1 observations
                for d in range(1, max_d + 1):
                    start = t - d + 1
                    # Calcul de la contribution des observations sur le segment [start, t]
                    if start == 0:
                        log_emis = cum_emission[j, t]
                        # Pas de segment précédent, utiliser l'initialisation
                        candidate = log_initial[j] + log_duration[j, d - 1] + log_emis
                    else:
                        log_emis = cum_emission[j, t] - cum_emission[j, start - 1]
                        # Somme sur les transitions depuis tous les états possibles à la fin du segment précédent
                        candidate = (np.logaddexp.reduce(log_alpha[start - 1, :] + log_transition[:, j])
                                     + log_duration[j, d - 1] + log_emis)
                    somme_d = np.logaddexp(somme_d, candidate)
                log_alpha[t, j] = somme_d
        
        log_prob = np.logaddexp.reduce(log_alpha[T - 1, :])
        return np.exp(log_prob)
# Exemple d'utilisation
if __name__ == "__main__":
    hsmm_model = HSMM("data/markov/fuji_long_year_4.toml")
    hsmm_model.display_parameters()
    states, observations = hsmm_model.generate_sequence(10, 20)
    print("\nGenerated Sequence of States:", states)
    print("Generated Sequence of Observations:", observations)
    
    prob_O = hsmm_model.forward_algorithm(observations)
    print("\nProbability of Observed Sequence:", prob_O)


