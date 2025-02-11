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
    
    def generate_sequence(self,nb_zones = 100):
            sequence_states = []
            sequence_observations = []
            
            # Initialisation
            current_state = np.random.choice(len(self.initial_probabilities), p=self.initial_probabilities)

            for _ in range(nb_zones):
                sequence_states.append(current_state)
                # Générer la durée pour cet état
                duration_probs = self.duration_matrix[current_state]
                duration_probs /= duration_probs.sum()  # Normalisation
                lower_bound = int(self.data['occupancy_distributions'][current_state]['bounds'][0])# ajout du décalage de distribution
                duration = np.random.choice(len(duration_probs), p=duration_probs) + lower_bound
                
                # Générer les observations pendant la durée
                for _ in range(duration):
                    obs_probs = self.observation_distributions[current_state]
                    observation = np.random.choice(len(obs_probs), p=obs_probs)
                    sequence_observations.append(observation)
                # Transition vers un nouvel état
                new_state = np.random.choice(len(self.transition_probabilities), p=self.transition_probabilities[current_state])
                if new_state == 6:  # par exemple, état absorbant
                    break
                current_state = new_state

            return sequence_states, sequence_observations
    def generate_bounded_sequence(self, l_bound, u_bound):
        
        sequence_states = None
        sequence_observations = None
        length = u_bound +1 # initialisation de la longueur de la séquence piur passer la boucle while
        count = 0 # compteur pour éviter les boucles infinies
        while length > u_bound or length < l_bound:
            sequence_states, sequence_observations = self.generate_sequence()
            count+=1
            length = len(sequence_observations)
            if count  == 1000:
                print("Impossible de générer une séquence dans les bornes demandées (trop d'itérations)")
                break      
        # print(sequence_observations) 
        sequence_observations = [''.join(str(d)) for d in sequence_observations]
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
        log_prob = log_prob   # Normalisation par la longueur de la séquence
        return np.exp(log_prob)
# Exemple d'utilisation
if __name__ == "__main__":
    hsmm_model = HSMM("data/markov/fuji_long_year_1.toml")
    import pandas as pd
    df = pd.read_csv("out/datasetcustom_comp_10000.csv")
    var_names = ["Sequence"]
    df = df[(df["Observation"] == "LARGE") & (df["Year"] == "Y1")]
    df = df[var_names]
    df.head()
    import pandas as pd
    df2 = pd.read_csv("out/sequence_analysis_dataset10000.csv")
    var_names = ["Sequence"]
    df2 = df2[(df2["Observation"] == "LARGE") & (df2["Year"] == "Y1")]
    df2 = df2[var_names]
    df2 = df2.head(1000)
    print(df2.head())
    pylist = []
    for seq_str in df2['Sequence']:
        seq = [int(char) for char in seq_str]  # Convertit chaque caractère en un entier et l'encapsule dans une liste  # Ajout du marqueur de fin de séquence
        pylist.append(seq)

    print(pylist[0])

    probs = []
    for i in tqdm(range(len(pylist))):
        prob = hsmm_model.forward_algorithm(pylist[i])
        prob = np.log(prob)/len(pylist[i])
        probs.append(prob)
    
    dfprobs = pd.DataFrame(probs,columns=["Probs"])
    dfprobs.to_csv("out/probs_dataset_sequenceanamysis_perso1000lignes.csv",index=False)




    # hsmm_model.display_parameters()
    # states, observations = hsmm_model.generate_sequence(10, 20)
    # print("\nGenerated Sequence of States:", states)
    # print("Generated Sequence of Observations:", observations)
    
    # prob_O = hsmm_model.forward_algorithm(observations)
    # print("\nProbability of Observed Sequence:", prob_O)


