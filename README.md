# 📊 Transaction Network Visualization App

Une application web interactive construite avec **Streamlit** pour visualiser et analyser les réseaux de transactions.

## 🎯 Description

Cette application permet de :
- **Visualiser** les réseaux de transactions sous forme de graphiques interactifs
- **Filtrer** les données par période (1 mois, 2 mois, 3 mois, 6 mois, 1 an, ou tout)
- **Analyser** les degrés de connexion entre les entités (1-2 liens, 2-5 liens, etc.)
- **Identifier** les patterns et clusters dans les données de transactions
- **Explorer** les relations entre les acteurs du réseau

## 🚀 Démarrage

### Prérequis
- Python 3.9+
- pip

### Installation

1. Clonez le repository
```bash
git clone <url-du-repository>
cd streamlit-app
```

2. Installez les dépendances
```bash
pip install -r src/requirements.txt
```

### Lancer l'application

```bash
streamlit run src/app.py
```

L'application s'ouvrira par défaut dans votre navigateur à l'adresse :
```
http://localhost:8501
```

## 🌐 Application déployée

L'application est disponible en ligne à l'adresse :
```
https://graph-transactions-relations.streamlit.app/
```

## 📦 Structure du projet

```
streamlit-app/
├── src/
│   ├── app.py              # Application principale Streamlit
│   └── requirements.txt     # Dépendances Python
├── data/
│   └── transactions/       # Données des transactions (format parquet)
├── scripts/
│   ├── generate_data.py    # Script pour générer les données
│   └── generate_structure.py
├── pyproject.toml          # Configuration du projet
└── README.md               # Ce fichier
```

## 🔧 Technologies utilisées

- **Streamlit** - Framework web pour la création d'applications data
- **Pandas** - Analyse et manipulation de données
- **NetworkX** - Création et analyse de graphes/réseaux
- **Plotly** - Visualisations interactives
- **Scikit-learn** - Machine learning (clustering, normalisation)
- **PyArrow** - Gestion des fichiers parquet

## 📊 Fonctionnalités principales

### Filtres temporels
- 1 mois
- 2 mois
- 3 mois
- 6 mois
- 1 an
- Tout (pas de filtre)

### Filtres par degrés de connexion
- Tous les nœuds
- 1-2 liens
- 2-5 liens
- 5-10 liens
- 10-20 liens
- 20-50 liens
- 50-100 liens
- Plus de 100 liens

## 💡 Utilisation

1. Lancez l'application avec la commande ci-dessus
2. Utilisez les filtres dans la barre latérale pour affiner votre analyse
3. Explorez le graphique interactif pour identifier les patterns
4. Analysez les clusters et les connexions principales

## 📝 Notes de développement

- Les données sont cachées en mémoire pour améliorer les performances
- Utilise Plotly pour les visualisations interactives
- Intégration de la détection de clusters avec K-Means

---

**Lien de l'application déployée** : https://graph-transactions-relations.streamlit.app/
