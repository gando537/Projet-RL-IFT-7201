# Apprentissage par renforcement sécuritaire – IFT-7201

Ce dépôt contient le code, l’analyse préliminaire et les expériences du projet de cours réalisé dans le cadre du cours **IFT-7201 – Apprentissage par renforcement** à l’Université Laval.

## Objectif du projet

L’objectif est d’étudier et d’évaluer des approches en **apprentissage par renforcement sécuritaire (Safe Reinforcement Learning)** dans un contexte industriel simulé, en se concentrant sur le **transfert de politique** depuis un environnement de simulation vers un environnement réel.

## Contenu du projet

- `analyse_preliminaire.tex` : Document LaTeX de l’analyse préliminaire (revue de littérature, formulation du problème, choix des méthodes).
- `biblio.bib` : Bibliographie au format BibTeX.
- `src/` : Code source des méthodes et des expériences.
- `notebooks/` : Cahiers Jupyter pour les tests exploratoires.
- `README.md` : Ce fichier.

## Problème étudié

Nous explorons une variante du RL sécuritaire centrée sur le **transfert de politiques entre simulation et réalité**, en évitant les états catastrophiques (récompenses très négatives) lors de l'exécution dans un environnement réel.

## Méthodes comparées

- **Méthode classique** : Deep Q-Network (DQN), à partir de la bibliothèque [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
- **Méthode sécuritaire** : Sim-to-Lab-to-Real, incluant un mécanisme de shielding pour bloquer les actions risquées lors du transfert.

## Librairies utilisées

- Python 3.10+
- `gymnasium`
- `stable-baselines3`
- `torch`
- `numpy`
- `matplotlib`
- `safeml` (si utilisé ou à définir)
- `jupyter` (pour les notebooks)

## Auteurs

- **Mouhamed Gando Diallo** — [mouhamed-gando.diallo.1@ulaval.ca](mailto:mouhamed-gando.diallo.1@ulaval.ca)  
- **Renaud Djekornonde Raoudel** — [renaud.djekornonde-raoudel.1@ulaval.ca](mailto:renaud.djekornonde-raoudel.1@ulaval.ca)

## Dates importantes

- Remise de l’analyse préliminaire : **31 mars 2025**
- Remise du rapport final : **2 mai 2025**

## Licence

Ce projet est à usage académique uniquement.
