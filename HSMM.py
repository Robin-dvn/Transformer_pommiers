import toml
import numpy as np
from hsmmlearn.hsmm import HSMMModel
from hsmmlearn.emissions import MultinomialEmissions
from scipy.stats import poisson, nbinom
from hsmmlearn.base import _fb_impl

class HSMMProcessor:
    def __init__(self, file_path):
        """Initialise le modèle HSMM à partir d'un fichier TOML."""
        self.file_path = file_path
        self.model = self.load_hsmm_from_toml()

    def load_hsmm_from_toml(self):
        """Charge un HSMM depuis un fichier TOML et construit le modèle avec hsmmlearn."""
        hsmm_data = toml.load(self.file_path)
        
        # Extraire les paramètres
        initial_probabilities = np.array(hsmm_data['initial_probabilities'])
        transition_probabilities = np.array(hsmm_data['transition_probabilities'])
        observation_distributions = np.array(hsmm_data['observation_distributions'])
        occupancy_distributions = hsmm_data['occupancy_distributions']
        
        # Définir les distributions d'émission
        self.n_states = len(initial_probabilities)
        emissions = MultinomialEmissions(probabilities=observation_distributions)
        
        
        # Définir les distributions de durée
        durations = []
        for occ in occupancy_distributions:
            if occ['distribution'] == 'NEGATIVE_BINOMIAL':
                durations.append(nbinom(n=occ['parameter'], p=occ['probability']))
            elif occ['distribution'] == 'POISSON':
                durations.append(poisson(mu=occ['parameter']))
            else:
                raise ValueError(f"Distribution de durée inconnue: {occ['distribution']}")
        # Compléter la liste des durées avec une distribution Poisson par défaut si besoin
        while len(durations) < 7:
            durations.append(poisson(mu=5))  # Par défaut, durée moyenne de 5

        
        # Créer le modèle HSMM
        model = HSMMModel(
            emissions=emissions,
            durations=durations,
            tmat=transition_probabilities,
            startprob=initial_probabilities,
            support_cutoff=10  # Durée max prise en compte
        )
        
        return model
    def sequence_probability(self, obs):
        print(obs.shape)
        print(obs)
        """Calcule la log-probabilité d'une séquence d'observations sous le HSMM."""
        obs = np.atleast_1d(obs)  # Assure que c'est un tableau 1D
        tau = len(obs)  # Longueur de la séquence
        j = self.model.n_states  # Nombre d'états cachés
        m = self.model.n_durations  # Nombre de durées possibles


        # Initialisation des matrices utilisées par _fb_impl()
        f = np.zeros((j, tau))
        l = np.zeros((j, tau))
        g = np.zeros((j, tau))
        l1 = np.zeros((j, tau))
        n = np.zeros(tau)  # Contiendra la probabilité finale
        norm = np.zeros((j, tau))
        eta = np.zeros((j, m))
        xi = np.zeros((j, m))
    

        # Calcul des probabilités d’émission P(s_t | q_t)
        likelihoods = self.model.emissions.likelihood(obs)
        likelihoods[likelihoods < 1e-12] = 1e-12  # Pour éviter les zéros log

        # Appel de _fb_impl() pour exécuter l'algorithme Forward-Backward
        err = _fb_impl(
            1, tau, j, m,
            self.model._durations_flat.copy(), self.model._tmat_flat.copy(), self.model._startprob.copy(),
            likelihoods.reshape(-1),
            f.reshape(-1), l.reshape(-1), g.reshape(-1), l1.reshape(-1),
            n, norm.reshape(-1), eta.reshape(-1), xi.reshape(-1)
        )

        if err != 0:
            raise RuntimeError("Erreur dans l'algorithme Forward-Backward.")

        return np.log(n).sum()  # Retourne la log-probabilité de la séquence
    

    def compute_sequence_probability(self, sequence):
        """Calcule la probabilité d'une séquence sous le modèle HSMM."""
        return self.sequence_probability(np.array(sequence))


# Exemple d'utilisation

if __name__ == "__main__":
    
    toml_file = "data/markov/fuji_long_year_4.toml"  # Remplace par ton fichier réel
    hsmm_processor = HSMMProcessor(toml_file)

    # Séquence d'observation donnée (remplace par une séquence réelle)
    sequence = [0, 3, 3, 4, 4, 5]  # Exemples de symboles observés

    log_prob = hsmm_processor.compute_sequence_probability(sequence)
    print(f"Log-probabilité de la séquence\n : {log_prob}")
