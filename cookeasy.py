import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pyautogui
import random
import time
import cv2
import osu_env
from osu_env import OsuEnv
import pandas as pd
from PIL import ImageGrab

game_params = {
    'screen_size': (800, 600),
    'num_actions': 2
}

total_missed_hits = 

missed_hits = total_missed_hits

score = 1000000 * (300 - missed_hits) / 300

actions = ['Z', 'X']

nombre_parties = int(input("Entrez le nombre de parties à générer : "))

image = ImageGrab.grab()

env = osu_env.OsuEnv(game_params)

observation, reward, done, info = env.step(actions, image)

# Fonction appelée lorsque le bot clique sur un cercle
def on_circle_click():
  global time_clicked
  time_clicked = time.time()

# Lorsque le bot clique sur un cercle, on appelle la fonction on_circle_click()
time_clicked = None
on_circle_click()

# Affiche le temps auquel le bot a cliqué sur le cercle (en secondes)
print(time_clicked)

# Récupérez le temps auquel le joueur a cliqué sur un cercle et le temps auquel ce cercle était censé être cliqué
time_clicked = time.time()  # Récupérez le temps auquel le joueur a cliqué sur un cercle
expected_time = time_clicked # Récupérez le temps auquel ce cercle était censé être cliqué

# Calculez la différence entre ces deux temps
difference = time_clicked - expected_time

# Définissez la précision en fonction de la différence
if difference < 0:
  precision = -difference / expected_time
else:
  precision = difference / expected_time

time_clicked = time.time()

affichage_du_cercle = time.time()

time_displayed = affichage_du_cercle

temps_de_reaction = time_clicked - time_displayed

# Charger les données de parties de osu!Standard à partir d'un fichier CSV
data = pd.read_csv('osu_data.csv')

# Récupérez les données de jeu à partir de parties de osu!Standard

# Sélectionnez les caractéristiques (features) à utiliser pour entraîner le modèle
x = data[['score', 'precision', 'response_time']]  # Utilisez le score, la précision et le temps de réponse comme features
donnees = {'precision': precision, 'temps de reaction': temps_de_reaction, 'score': score}

# Sélectionnez la cible (target) à prédire avec le modèle
y = data['action']  # Utilisez l'action prise comme cible à prédire

# Définissez le nombre de couches et de neurones de votre modèle
nombre_couches = 4
nombre_neurones = 168

# Séparer vos données en ensembles d'entraînement et de validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

# Transformer vos données en tableau NumPy
x_train = np.array(x_train)
y_val = np.array(y_val)

# Vérifier que x_train et x_val ont la bonne forme
print(x_train.shape) # Doit afficher (n_échantillons, n_caractéristiques)
print(x_val.shape) # Doit afficher (n_échantillons, n_caractéristiques)

# Définissez la fonction de perte et l'optimiseur à utiliser
perte = 'categorical_crossentropy'
optimiseur = tf.keras.optimizers.Adam()

# Créez un modèle vide
modele = keras.Sequential()

# Ajoutez des couches de neurones au modèle
modele.add(keras.layers.Dense(nombre_neurones, input_shape=(x_train.shape[1],), activation='relu'))
for _ in range(nombre_couches - 1):
  modele.add(keras.layers.Dense(nombre_neurones, activation='relu'))

modele.compile(loss=perte, optimizer=optimiseur, metrics=['accuracy'])

# Définissez le nombre d'époques et de batchs à utiliser pour l'entraînement
epoques = 10
batch_taille = 32

last_action_was_z = False

def choose_action():
  global last_action_was_z
  if last_action_was_z:
    last_action_was_z = False
    return 'X'
  else:
    last_action_was_z = True
    return 'Z'

# Entraînez le modèle
historique = modele.fit(x_train, y_train, epochs=epoques, batch_size=batch_taille, validation_data=(x_val, y_val))

# Écrivez une fonction de simulation qui génère des parties de osu!Standard
def osu_data(nombre_parties):
    donnees = []
    for _ in range(nombre_parties):
        partie = PartieDeOsu()
        while not partie.partie_terminee:
            # Générez une action aléatoire pour la partie en cours
            action = random.choice(partie.actions_possibles)
            # Mettez à jour l'état de la partie et récupérez les données de cette action
            score, precision, response_time = partie.update_state(action)
            donnees.append({'score': score, 'precision': precision, 'response_time': response_time, 'action': action})
    return donnees

data = osu_data(100) # Générez 100 parties de osu!Standard pour entraîner le modèle

class PartieDeOsu:
    def __init__(self):
        # Initialisez les variables de la classe ici
        self.partie_terminee = False
        self.actions_possibles = ['Z', 'X']
        
    def choisir_action(self, partie):
        # Utilisez l'état de la partie pour choisir une action à effectuer
        if partie.x > 50:
            return 'Z'
        else:
            return 'X'
    def actions_possibles(self):
        # Retournez la liste des actions possibles
        return random.choice(self.actions_possibles)
    
    def partie_terminee(self):
        # Mettez à jour l'état de la partie et retournez True si elle est terminée, False sinon
        if self.partie_terminee:
            return True
        else:
            return False

# Initialisez une partie de osu!
partie = PartieDeOsu()

# Choisissez une action à l'aide de la fonction choisir_action
action = partie.choisir_action(partie)

# Affichez la liste des actions possibles
print(partie.action_possibles())
            
env.get_image()
env.render(mode='video')

donnees = []

for i in range(nombre_parties):
    # Créez une nouvelle partie
    partie = PartieDeOsu()

    # Initialisez une liste pour stocker les actions de la partie
    actions_partie = []

    # Jouez la partie jusqu'à ce qu'elle soit terminée
    while not partie.partie_terminee():
        # Choisissez une action à effectuer
        action = partie.choisir_action(partie)
        # Ajoutez l'action à la liste des actions de la partie
        actions_partie.append(action)
        # Exécutez l'action dans l'environnement de simulation
        observation, reward, done, info = env.step(action, image)
        # Mettez à jour l'état de la partie
        partie.update_state(observation)
        
    # Ajoutez les actions de la partie à la liste des données
    donnees.append(actions_partie)