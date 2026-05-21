# Sommaire
1. [Mots-clés](#mots-clés)
2. [I. Introduction à MLflow](#i-introduction-à-mlflow)
3. [II. Les Quatre Composants Fondamentaux (Piliers)](#ii-les-quatre-composants-fondamentaux-piliers)
   * [A. MLflow Tracking (Suivi des expériences)](#a-mlflow-tracking-suivi-des-expériences)
   * [B. MLflow Projects (Packaging)](#b-mlflow-projects-packaging)
   * [C. MLflow Models (Format de déploiement)](#c-mlflow-models-format-de-déploiement)
   * [D. MLflow Model Registry (Gestion des versions)](#d-mlflow-model-registry-gestion-des-versions)
4. [III. Étendue de l'utilisation et Bénéfices Académiques](#iii-étendue-de-lutilisation-et-bénéfices-académiques)

---

### Mots-clés
* MLflow
* Cycle de vie du Machine Learning (Lifecycle)
* Reproductibilité
* Artefacts (Artifacts)
* Versionnement de modèles

---

### I. Introduction à MLflow

[cite_start]MLflow est une plateforme open-source essentielle pour la gestion de bout en bout du cycle de vie de l'apprentissage automatique[cite: 25, 72]. [cite_start]L'intégration de cet outil répond à une problématique majeure en Data Science : la dispersion des résultats et la difficulté de reproduire les modèles[cite: 26]. [cite_start]En centralisant le suivi, le packaging et le versionnement, MLflow standardise les flux de travail et facilite la comparaison objective des performances des algorithmes[cite: 26].

---

### II. Les Quatre Composants Fondamentaux (Piliers)



#### A. MLflow Tracking (Suivi des expériences)
[cite_start]Ce composant agit comme un journal de bord automatisé pour l'ingénieur en Machine Learning[cite: 26, 40]. [cite_start]Il permet de consigner et de requêter les éléments clés lors de chaque exécution de code (*run*)[cite: 40].
* **Éléments enregistrés :**
  * [cite_start]**Paramètres :** Clés-valeurs de configuration (ex: nombre d'arbres pour un `Random Forest`, taux d'apprentissage)[cite: 30, 40].
  * [cite_start]**Métriques :** Évolutions des performances numériques (ex: *Accuracy*, *Log Loss*)[cite: 35, 40].
  * [cite_start]**Artefacts :** Fichiers lourds de sortie (matrices de confusion sérialisées, graphiques d'analyse exploratoire, fichiers de poids du modèle)[cite: 24, 40].
* **Application au projet :** [cite_start]Initialisation de l'expérience globale sous le nom `experiment_water_quality` sur un serveur local écoutant sur le port 5000[cite: 37, 39].

#### B. MLflow Projects (Packaging)
Ce module définit un format d'empaquetage standardisé pour organiser et décrire le code de code source afin qu'il puisse être exécuté de manière identique sur n'importe quelle plateforme.
* [cite_start]**Mécanisme :** Utilisation d'un fichier de configuration (généralement `MLproject`) à la racine du répertoire[cite: 58]. [cite_start]Ce fichier spécifie les dépendances logicielles (via un environnement Conda, Virtualenv ou un conteneur Docker) et les points d'entrée des scripts (ex: `python experiment.py`)[cite: 62].
* [cite_start]**Bénéfice :** Garantie de reproductibilité technique stricte pour l'évaluation[cite: 26, 55].

#### C. MLflow Models (Format de déploiement)
[cite_start]Un "MLflow Model" est un format standard de packaging qui permet d'utiliser le modèle entraîné dans divers environnements en aval[cite: 41, 44].
* **Mécanisme :** Chaque modèle est sauvegardé dans un dossier contenant le modèle sérialisé (ex: fichier `.pkl` ou `.onnx`) accompagné d'un fichier de configuration `MLmodel`. Ce fichier décrit les différentes "saveurs" (*flavors*) sous lesquelles le modèle peut être interprété (ex: saveur `python_function` ou saveur `sklearn`).
* **Bénéfice :** [cite_start]Flexibilité totale pour l'étape suivante, facilitant le déploiement direct dans une application Flask (`app.py`) via une API REST[cite: 44, 63].

#### D. MLflow Model Registry (Gestion des versions)
[cite_start]Le registre de modèles centralisé offre une interface et un ensemble d'API pour gérer collaborativement le cycle de vie complet des modèles d'une organisation[cite: 43, 71].
* **Fonctionnalités :**
  * [cite_start]Traçabilité complète de la lignée du modèle (quel *run* de quel projet a généré ce modèle)[cite: 26, 40].
  * [cite_start]Versionnement automatique (Version 1, Version 2, etc.)[cite: 26, 43].
  * [cite_start]Gestion des transitions d'états : passage du statut expérimental (*Staging*) au statut de production (*Production*)[cite: 43].

---

### III. Étendue de l'utilisation et Bénéfices Académiques

Dans le cadre d'un projet d'ingénierie IA, l'adoption de MLflow apporte une rigueur scientifique indispensable :

* [cite_start]**Élimination de l'effet "boîte noire" :** L'interface utilisateur de MLflow (`MLflow UI`) offre une visualisation claire des courbes d'apprentissage, facilitant l'analyse comparative fine entre les approches classiques (XGBoost) et le deep learning (Perceptron multicouches)[cite: 30, 34, 42].
* [cite_start]**Standardisation du livrable :** L'outil force le développeur à isoler proprement la phase de recherche expérimentale (`experiment.py`) de la phase applicative opérationnelle (`app.py`), une structure attendue dans les exigences de production industrielle[cite: 62, 63].