import toml
import numpy as np
from scipy.stats import nbinom, poisson,binom
from tqdm import tqdm

def generate_duration_matrix(toml_file, prob_cutoff=1e-20):
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
                assert prob != None,"La probabilité doit être spécifiée pour une distribution négative binomiale"
                d_max_auto = 1
                while nbinom.sf(d_max_auto, param, prob) > prob_cutoff:
                    #tire la probabilité de la distribution négative binomiale jusqu'à ce qu'elle soit inférieure au seuil de probabilité
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
    D = np.zeros((N+1, Tmax)) # +1 pour la distribution de l'état absorbant
    
    # Remplir la matrice avec les distributions
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
            probs = binom.pmf(durations, sup-inf, p) # gestion des bornes de la distribution (sup -inf)= nombres de tirages  
        elif dist['distribution'] == 'POISSON':
            param = dist['parameter']
            durations = np.arange(0, Tmax )
            probs = poisson.pmf(durations, param)
        
        else:
            raise ValueError(f"Distribution non supportée : {dist['distribution']}")
        
    
        # Remplir la matrice
        D[j, :len(probs)] = probs

        # Ajouter une ligne pour l'état absorbant
        D[6] = np.zeros(Tmax)  # Initialisation à 0
        D[6][-1] = 1.0  # Toute la probabilité sur Tmax

    
    return D



class HSMM:
    def __init__(self, toml_file):
        """
        Initialise un HSMM à partir d'un fichier TOML.
        """
        self.data = toml.load(toml_file)
        self.initial_probabilities = np.array(self.data['initial_probabilities'])
        self.transition_probabilities = np.array(self.data['transition_probabilities'])
        self.observation_distributions = np.array(self.data['observation_distributions'])

        # Normalisation des probabilités pour avoir une somme à 1
        self.observation_distributions /= self.observation_distributions.sum(axis=1, keepdims=True)
        self.transition_probabilities /= self.transition_probabilities.sum(axis=1, keepdims=True)
        self.initial_probabilities /= self.initial_probabilities.sum(axis=0, keepdims=True)

        # Génère la matrice de distribution de durée 
        self.duration_matrix = generate_duration_matrix(toml_file)
    
    def get_initial_probabilities(self):
        """
        Retourne les probabilités initiales du modèle HSMM.
        
        Retourne :
        - initial_probabilities : tableau des probabilités initiales
        """
        return self.initial_probabilities
    
    def get_transition_matrix(self):
        """
        Retourne la matrice de transition du modèle HSMM.
        
        Retourne :
        - transition_probabilities : matrice des probabilités de transition
        """
        return self.transition_probabilities
    
    def get_observation_matrix(self):
        """
        Retourne la matrice des distributions d'observation du modèle HSMM.
        
        Retourne :
        - observation_distributions : matrice des distributions d'observation
        """
        return self.observation_distributions
    
    def get_duration_matrix(self):
        """
        Retourne la matrice des distributions de durée du modèle HSMM.
        
        Retourne :
        - duration_matrix : matrice des distributions de durée
        """
        return self.duration_matrix
    
    def display_parameters(self):
        """
        Affiche les paramètres du modèle HSMM, y compris les probabilités initiales,
        les probabilités de transition, les distributions d'observation et la matrice de durée.
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
        Génère une séquence d'états et d'observations pour un nombre donné de zones.

        Paramètres :
        - nb_zones : nombre de zones à générer (par défaut 100)

        Retourne :
        - sequence_states : liste des états générés
        - sequence_observations : liste des observations générées
        """
        sequence_states = []
        sequence_observations = []

        # Initialisation
        current_state = np.random.choice(len(self.initial_probabilities), p=self.initial_probabilities)

        for _ in range(nb_zones):
            sequence_states.append(current_state)
            # Générer la durée pour cet état
            duration_probs = self.duration_matrix[current_state]
            duration_probs /= duration_probs.sum()  # Normalisation pour sommer à 1
            lower_bound = int(self.data['occupancy_distributions'][current_state]['bounds'][0])  # ajout du décalage de distribution
            duration = np.random.choice(len(duration_probs), p=duration_probs) + lower_bound

            # Générer les observations pendant la durée
            for _ in range(duration):
                obs_probs = self.observation_distributions[current_state]
                observation = np.random.choice(len(obs_probs), p=obs_probs)
                sequence_observations.append(observation)

            # Transition vers un nouvel état
            new_state = np.random.choice(len(self.transition_probabilities), p=self.transition_probabilities[current_state])
            if new_state == 6:  # stop génération à l'état absorbant
                break
            current_state = new_state

        return sequence_states, sequence_observations

    def generate_bounded_sequence(self, l_bound, u_bound):
        """
        Génère une séquence d'états et d'observations dont la longueur est comprise entre des bornes spécifiées.

        Paramètres :
        - l_bound : borne inférieure de la longueur de la séquence
        - u_bound : borne supérieure de la longueur de la séquence

        Retourne :
        - sequence_states : liste des états générés
        - sequence_observations : liste des observations générées
        """
        sequence_states = None
        sequence_observations = None
        length = u_bound + 1  # initialisation de la longueur de la séquence pour passer la boucle while
        count = 0  # compteur pour éviter les boucles infinies

        # génère une séquence et réitère si la longueur n'est pas dans les bornes
        while length > u_bound or length < l_bound:
            sequence_states, sequence_observations = self.generate_sequence()
            count += 1
            length = len(sequence_observations)
            if count == 1000:
                print("Impossible de générer une séquence dans les bornes demandées (trop d'itérations)")
                break

        sequence_observations = [''.join(str(d)) for d in sequence_observations]

        return sequence_states, sequence_observations
    
    def forward_algorithm(self, observations):
        """
        Applique l'algorithme Forward en log-espace pour un HSMM, en sommant sur toutes les durées possibles.
        Cette implémentation calcule la probabilité que la séquence d'observations soit générée par le modèle.

        Paramètres :
        - observations : liste des observations pour lesquelles calculer la probabilité

        Retourne :
        - probabilité : probabilité que la séquence d'observations soit générée par le modèle
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
                if j != N-1:
                    lower_bound = int(self.data['occupancy_distributions'][j]['bounds'][0])
                else:
                    lower_bound = 0
                d_max_possible = min(t + 1, lower_bound + D_max - 1)
                for d in range(lower_bound, d_max_possible + 1):
                    start = t - d + 1
                    # Calcul de la contribution des observations sur le segment [start, t]
                    dur_idx = d - lower_bound
                    if start == 0:
                        log_emis = cum_emission[j, t]
                        # Pas de segment précédent, utiliser l'initialisation
                        candidate = log_initial[j] + log_duration[j, dur_idx] + log_emis
                    else:
                        log_emis = cum_emission[j, t] - cum_emission[j, start - 1]
                        # Somme sur les transitions depuis tous les états possibles à la fin du segment précédent
                        candidate = (np.logaddexp.reduce(log_alpha[start - 1, :] + log_transition[:, j])
                                     + log_duration[j, dur_idx] + log_emis)
                    somme_d = np.logaddexp(somme_d, candidate)
                log_alpha[t, j] = somme_d
        
        log_prob = np.logaddexp.reduce(log_alpha[T - 1, :])
        log_prob = log_prob /T  # Normalisation par la longueur de la séquence
        return np.exp(log_prob)


# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd

    # Initialisation du modèle HSMM à partir d'un fichier TOML
    hsmm_model = HSMM("data/markov/fuji_long_year_1.toml")

    # Exemple d'utilisation de display_parameters
    print("Affichage des paramètres du modèle HSMM :")
    hsmm_model.display_parameters()

    # Exemple de génération de séquence bornée
    print("\nGénération d'une séquence d'états et d'observations bornée :")
    bounded_states, bounded_observations = hsmm_model.generate_bounded_sequence(5, 15)
    print("Generated Bounded Sequence of States:", bounded_states)
    print("Generated Bounded Sequence of Observations:", bounded_observations)
    sequence_observations = [int(d) for d in bounded_observations]
    print(sequence_observations)
    # Exemple d'utilisation de l'algorithme Forward
    print("\nCalcul de la probabilité d'une séquence d'observations avec l'algorithme Forward :")
    prob_O = hsmm_model.forward_algorithm(sequence_observations)
    print("Normalized Probability of Observed Sequence:", prob_O)

    # Exemple d'analyse de séquence (calcul de propbabilitées) à partir d'un fichier CSV
    analyse = True
    if analyse:
        print("\nAnalyse de séquence à partir d'un fichier CSV :")
        df = pd.read_csv("out/sequence_analysis_dataset10000.csv")
        df = df[(df["Observation"] == "LARGE") & (df["Year"] == "Y1")]
        # df = df.head(1000)

        pylist = []
        for seq_str in df['Sequence']:
            seq = [int(char) for char in seq_str]  # Convertit chaque caractère en un entier et l'encapsule dans une liste
            pylist.append(seq)

        probs = []
        for i in tqdm(range(len(pylist))):
            prob = hsmm_model.forward_algorithm(pylist[i])
            probs.append(np.log(prob))

        dfprobs = pd.DataFrame(probs, columns=["Probs"])
        dfprobs.to_csv("out/probs_dataset_sequence_analysis_perso_corrige_a_garder.csv", index=False)
