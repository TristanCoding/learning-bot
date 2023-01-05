from msilib.schema import SelfReg
from typing import Self
import gym
from gym import spaces
import pyautogui
import cv2
import numpy as np
import math
from PIL import ImageGrab
import time
import os
import subprocess

class OsuEnv(gym.Env):
    def __init__(self, game_params):
        self.game_params = game_params # Stocker les paramètres de jeu en tant que variable d'instance
        self.response_time += 0
        # Définissez l'espace d'observation de l'environnement
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.game_params['screen_size'][0], self.game_params['screen_size'][1], 3))
        # Définissez l'espace d'action de l'environnement
        self.action_space = spaces.Discrete(self.game_params['num_actions'])
        # Définissez une liste de cercles qui doit être cliquée dans l'ordre
        self.circles_to_click = [(100, 100), (200, 200), (300, 300)]

    def step(self, action, image):
        # Récupérez l'image de l'environnement
        image = self.get_image()

        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détectez les cercles dans l'image
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

        # Récupérez la position de la souris
        mouse_x, mouse_y = pyautogui.position()
        
        # Récupérez le cercle actuellement sélectionné dans la liste de cercles à cliquer
        selected_circle = self.circles_to_click[self.current_circle_index]

        # Si le cercle le plus proche de la souris est égal au cercle sélectionné, cliquez dessus
        if closest_circle == selected_circle:
            pyautogui.click(closest_circle[0], closest_circle[1])
            self.current_circle_index += 1
        
        # Trouvez le cercle le plus proche de la souris
        closest_circle = None
        closest_distance = float("inf")
        for circle in circles[0, :]:
            # Calculez la distance entre le cercle et la souris
            distance = math.sqrt((circle[0] - mouse_x)**2 + (circle[1] - mouse_y)**2)
            if distance < closest_distance:
                closest_circle = circle
                closest_distance = distance

        # Cliquez sur le centre du cercle le plus proche de la souris
        pyautogui.click(closest_circle[0], closest_circle[1])

        # Implémentez la logique pour exécuter l'action dans l'environnement de simulation
        # Retournez l'observation, le reward, done et info
        observation = self.get_observation(self)
        reward = self.get_reward(self)
        done = self.is_done(self)
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        # Implémentez la logique pour afficher l'environnement de simulation à l'écran (facultatif)
        pass  # ajoutez cette ligne pour éviter une erreur de syntaxe
        
    def get_image(self):
     # Capturez l'image de l'environnement
     image = ImageGrab.grab()
     # Convertir l'image en un tableau NumPy
     image_np = np.array(image)
     # Convertir l'image en un format compatible avec OpenCV
     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
     return image_np
    
def screen_record():
        last_time = time.time()
        while(True):
            # Capture l'écran
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imshow('screen', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

def close(self):
    # Ferme osu!
    subprocess.run(["taskkill", "/f", "/im", "nom_du_processus.exe"])

def preprocess_image(self, image):
    # Convertir l'image en niveaux de gris
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Appliquer un filtre Gaussien pour lisser l'image
    processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
    # Réduire le bruit en utilisant un seuillage adaptatif
    processed_image = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed_image

def get_observation(self):
    # Implémentez la logique pour récupérer les informations observables de l'environnement de simulation
    # Par exemple, vous pouvez récupérer la position des objets du jeu, le score actuel, le nombre de vies restantes, etc.
    obs = [self.current_score, self.current_lives, self.player_x, self.player_y]
    return obs

def get_reward(self):
    # Récupérez les paramètres de jeu
    difficulty = self.game_params['difficulty']
    target_response_time = self.game_params['target_response_time']
    agent_level = self.game_params['agent_level']

    # Initialisez la récompense à 0
    reward = 0

    # Si le joueur a réalisé un bon coup, ajoutez une récompense positive
    if self.current_score > self.previous_score:
        reward += 1

    # Si le joueur n'a plus de vies, ajoutez une récompense négative
    if self.current_lives == 0:
        reward -= 1

    # Si le temps de réponse de l'agent est inférieur au temps de réponse cible, ajoutez une récompense positive
    if self.current_response_time < target_response_time:
        reward += 1

    # Utilisez la difficulté de la carte pour ajuster la récompense
    if difficulty == 'easy':
        reward *= 0.5
    elif difficulty == 'medium':
        reward *= 1
    elif difficulty == 'hard':
        reward *= 1.5

    # Utilisez le niveau de jeu de l'agent pour ajuster la récompense
    if agent_level == 'beginner':
        reward *= 0.5
    elif agent_level == 'intermediate':
        reward *= 1
    elif agent_level == 'expert':
        reward *= 1.5

    return reward

def is_done(self):
    # Utiliser le niveau de jeu de l'agent et le temps de réponse cible pour déterminer si la partie est terminée
    level = self.game_params['level']
    target_response_time = self.game_params['target_response_time']
    if level > 10 or self.response_time > target_response_time:
        return True

    # Implémentez la logique pour déterminer si la partie de jeu est terminée en fonction de l'état actuel de l'environnement de simulation
    if self.current_lives == 0:
        # Si le joueur n'a plus de vies, la partie est terminée
        return True
    elif self.current_score == self.max_score:
        # Si le joueur a atteint le score maximum, la partie est terminée
        return True
    else:
        # Dans tous les autres cas, la partie n'est pas terminée
        return False
    
def reset(self):
    # Réinitialisez les paramètres internes de l'environnement
    self.current_score = 0
    self.current_lives = 3
    self.player_x = 0
    self.player_y = 0
    self.current_circle_index += 0
    # Capturez l'image initiale de l'environnement
    image = self.get_image()
    # Traitez l'image et retournez l'observation initiale
    processed_image = self.preprocess_image(image)
    return processed_image

circles_positions = [] # Liste qui stockera les positions des cercles dans l'ordre
current_circle_index = 0 # Compteur qui indique sur quel cercle le bot doit cliquer

image = ImageGrab.grab()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détectez les cercles dans l'image
circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Récupérez la position de la souris
mouse_x, mouse_y = pyautogui.position()

# Trouvez le cercle le plus proche de la souris
closest_circle = None
closest_distance = float("inf")
for circle in circles[0, :]:
    # Calculez la distance entre le cercle et la souris
    distance = math.sqrt((circle[0] - mouse_x)**2 + (circle[1] - mouse_y)**2)
    if distance < closest_distance:
        closest_circle = circle
        closest_distance = distance

# Vérifiez si le cercle le plus proche de la souris est le cercle sur lequel le bot doit cliquer
if closest_circle == circles_positions[current_circle_index]:
    # Cliquez sur le centre du cercle le plus proche de la souris
    pyautogui.click(closest_circle[0], closest_circle[1])
    current_circle_index += 1 # Incrémentez le compteur
else:
    # Ne faites rien, le bot doit attendre de trouver le cercle suivant
    pass
