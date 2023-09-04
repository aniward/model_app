
import streamlit as st
import pandas as pd
import plotly_express as px
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
import re

# Définir le style CSS en utilisant Markdown pour la couleur de fond et le titre
 #Appliquer le style CSS pour le fond de la page


page_style = """
<style>
body {
  background-color: #f7f7f7;  /* Couleur de fond de page */
  padding-top: 0;  /* Pas d'espace en haut de la page */
}
h1 {
  margin-top: 0px;  /* Espace réduit au-dessus du titre */
  text-align: center;  /* Centrer le titre */
  font-size: 24px;  /* Taille de police du titre */
}
.prediction-value {
  font-size: 20px;  /* Taille de police du texte Predicted Value */
  color: blue;  /* Couleur bleue pour le texte Predicted Value */
}
  .total-sum {
  font-size: 20px;  /* Taille de police du texte Predicted Value */
  color: blue;  /* Couleur bleue pour le texte Predicted Value */
}
</style>
"""

#--------------------------------- Head
## ===> Header
#st.title("")
st.markdown("<h1 style='text-align: center;'> Prédiction de la note d'un client</h1>", unsafe_allow_html=True)
#st.title("Stock Value Prediction using Neural Networks")
col1, col2 = st.columns([0.15, 0.15])

col1.image("https://logowik.com/content/uploads/images/t_trustpilot2525.jpg")

col2.image("https://www.ikea.com/fr/fr/images/products/vagstranda-matelas-a-ressorts-ensaches-mi-ferme-bleu-clair__1150861_pe884901_s5.jpg?f=xl")

# st.markdown("""
            
# Ce projet a pour but de recueillir par webscrapping des données historiques d'entreprises cotées et puis de faire une analyse predictive in-sample en utilisant les Reccurent Neural Networks RNNs.""")




# Afficher le style en utilisant st.markdown()
st.markdown(page_style, unsafe_allow_html=True)

# Chemin vers votre image
# image_path = 'images.jpeg'

# # Insérer la balise HTML <img> pour afficher l'image
# st.markdown(f'<img src="{image_path}" alt="En-tête de page" style="max-width: 100%;">', unsafe_allow_html=True)




# Afficher le titre de la page en utilisant Markdown
#st.markdown("# Application Web permettant de prédire la note d'un client")


## Appelle du modèle retenu afin de noter les nouveau commentaire

def find_mot(tweet, mot):
    r = re.compile(mot)
    occurences = r.findall(tweet)
    return len(occurences)

mots_a_rechercher = ['pas', 'aucune', 'remboursement', 'bien', 'rapide', 'qualité', 'jours', 'recommande', 'parfait', 'bémol', 'réveil', 
                     'déchiré', 'médiocre', 'toujours', 'déçus','top', 'disant', 'bien', 'récupéré', 'meuble', 
                     'assemblage', 'conforme', 'similaire', 'jamais', 'confortable', 'écrase', 'faut', 
                     'maintient', 'compenser', 'catastrophique', 'arrivait', 'chers', 'cher', 'non', 
                     'auparavant', 'tiroirs', 'promettez', 'véritable', 'déçue', 'réglables', 'table', 'aime', 'coutures'
                     'convenaient', 'convenaient', 'semaines', 'merci', 'fond', 'larronde', 'tiendra', 'local', 
                     'insatisfait', 'réalistes', 'ressentons', 'louange', 'attends', 'express', 'dommage', 'sommaire',
                     'ressors', 'tissus', 'invérifiables', 'dérouler', 'différents', 'lenteur', 'débitée', 'toujours', 'commande', 
                     'peine', 'pendant', 'très', 'mois', 'car', 'disant', 'peu', 'nouvelle', 'mauvaise', 'donc', 'bois', 'rembourser', 'attentes',
                     'service client', 'livraison rapide', 'très bien', 'très bon', 'bonne qualité', 'qualité prix', 'rapport qualité', 'très bonne',
                     'très satisfaite', 'produit conforme', 'conforme description', 'délai livraison', 'bon rapport', 'très satisfait', 'très confortable',
                     'conforme attentes', 'très rapide', 'rapport qualité prix', 'bon rapport qualité', 'très bonne qualité', 'livraison très rapide',
                     'très bon matelas', 'service après vente', 'matelas très confortable', 'très bon produit','très bon rapport', 'ivraison rapide produit',
                     'plus mal dos', 'produit conforme attente', 'livraison rapide matelas', 'matelas conforme description', 'matelas très bonne', 'excellent rapport qualité',
                     'jours plus tard', 'service client très', 'bon rapport qualité prix', 'très bon rapport qualité', 'matelas très bonne qualité',
                     'excellent rapport qualité prix', 'rapport qualité prix livraison', 'livraison rapide produit conforme','tout très bien passé',
                     'impossible joindre service client', 'qualité prix livraison rapide', 'livraison très rapide matelas', 'rapport qualité prix recommande',
                     'produit très bien emballé','délai livraison peu long', 'passée', 'mail', 'étoiles', 'hier', 'réussi', 'référence', 'déception', 'fuir', 
                     'assez', 'frais', 'seule', 'longue', 'rien', 'trop','déçu', 'complément', 'réalité','vivement','ailleurs','facile', 'fermeté','petit','réactivité', 'commode', 'difficultés', 'dépannage', 'dépassé']


mots_a_rechercher2 = ['passée', 'mail', 'étoiles', 'hier', 'réussi', 'référence', 'déception', 'fuir', 'assez', 'frais', 'seule', 'longue', 'rien', 'trop','déçu', 
                     'complément', 'réalité','vivement','ailleurs','facile', 'fermeté','petit','réactivité', 'commode', 'difficultés', 'dépannage', 'dépassé']


# Charger les données prétraitées
df_preprocessed = pd.read_csv(r'C:/Users/FADOLO/Documents/GitHub/SupPY_Chain/data/df.csv')  # Remplacez 'chemin_vers_votre_fichier.csv' par le chemin réel de votre fichier

#Ajout des mots suppléme,taires à la table initiale utilisée pour ma modélisation
for mot in mots_a_rechercher2:
    df_preprocessed[mot] = df_preprocessed.commentaires.apply(lambda x: find_mot(str(x), mot))

# Préparer les données pour l'entraînement du modèle : Regroupement des modalites
df_preprocessed["Note_Client"]=df_preprocessed["nb_etoiles"].replace((1,2,3,4,5),(0,1,1,2,2))

##Echantillon dapprentissage et de test
y=df_preprocessed['Note_Client']
X=df_preprocessed.drop(['commentaires', 'entreprises', 'nb_etoiles', 'dates_comment', 'Sentiment','Note_Client'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

##Utilisation du Undersampling
rUs = RandomUnderSampler(random_state=1234)
X_ru, y_ru = rUs.fit_resample(X_train, y_train)

# Entraîner le modèle
clf = LogisticRegression(random_state=22, C=0.5623413251903491, solver='liblinear')
#final_clf = gridcvs['LogisticRegression']
clf.fit(X_ru, y_ru)

# Calcul et affichage de la matrice de confusion
y_pred=clf.predict(X_test)
confusion_matrix_eq = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
# print(confusion_matrix_eq)

# Calcul et affichage de la matrice de confusion
st.write("Matrice de confusion du modèle :")
confusion_matrix_eq.index.name = 'Classe réelle'
confusion_matrix_eq.columns.name = 'Classe prédite'
st.write(confusion_matrix_eq)

n_rows_to_display = 20
df_for_predictions = df_preprocessed.copy()
df_for_predictions['predicted'] = False
# fig = px.line(df_for_predictions.tail(n_rows_to_display), x='commentaires', y='Note_Client', color='predicted')
# online_ts_chart = st.plotly_chart(fig)

# Ajouter des informations supplémentaires à l'interface utilisateur Streamlit
new_row_info = st.empty()
predicted_row_warning = st.empty()

##Fonctions permettant de génerer et predire le nouveau commentaire
import numpy as np

def generate_new_row(df_for_predictions, user_comment):
    new_row = pd.DataFrame({
        'commentaires': [user_comment]
    })

    # Appliquer la fonction find_mot à chaque mot de mots_a_rechercher
    for mot in mots_a_rechercher:
        new_row[mot] = [find_mot(user_comment, mot)]

    return new_row.drop(columns=['commentaires'])  # Supprimer la colonne 'commentaires' pour la prédiction

def generate_new_prediction(df_for_predictions, new_row, clf):
    # utilisation du modèle (clf) pour faire une prédiction basée sur la nouvelle ligne.
    new_prediction = pd.DataFrame({
        'Note_Client': clf.predict(new_row)
    })
    return new_prediction

def add_row(df_for_predictions, new_prediction):
    # Ajouter la nouvelle prédiction à votre DataFrame
    new_prediction['commentaires'] = user_comment  # Utilisation du commentaire saisi
    df_for_predictions = pd.concat([df_for_predictions, new_prediction], ignore_index=True)

    return df_for_predictions

def animate(df, y_column, chart):
    # Mettez à jour le graphique interactif avec les nouvelles données
    fig = px.line(df, x='commentaires', y=y_column, color='predicted')
    chart.plotly_chart(fig)


if st.sidebar.checkbox('Prediction du nouveau commentaire'):
    # Ajoutez une zone de texte pour le commentaire de l'utilisateur
    user_comment = st.text_area("Entrer un commentaire pour la prediction", "")

    # Ajoutez un bouton pour effectuer la prédiction
    if st.button("Prediction"):

        # Convertir user_comment en minuscules
        user_comment = user_comment.lower()

        # Générer une nouvelle ligne pour la prédiction
        new_row = generate_new_row(df_for_predictions, user_comment)

        # Effectuer la prédiction sur la nouvelle ligne
        new_prediction = generate_new_prediction(df_for_predictions, new_row, clf)
        
        # Obtenir la valeur de prédiction
        prediction_value = new_prediction['Note_Client'].values[0]

        total_sum = new_row.sum().sum()

    # Vérifier si total_sum est égal à 0
        if total_sum == 0:
            st.warning("Précisez votre commentaire s'il vous plait.")
        else:
             
            # Créer une mise en page en colonnes pour les résultats
             col1, col2 = st.columns(2)

             # Afficher Valeur de la prédiction
             with col1:
                st.markdown("<div style='color: blue; font-size: 18px;'>Valeur de la prédiction</div>", unsafe_allow_html=True)
                #st.write(prediction_value)
                # Afficher Valeur de la prédiction avec le style personnalisé
                st.markdown(f'<p class="prediction-value">{prediction_value}</p>', unsafe_allow_html=True)
        

            # Afficher Total Sum of New Row
             with col2:
               st.markdown("<div style='color: blue; font-size: 18px;'>Nbre de variables utilisées dans la prédiction</div>", unsafe_allow_html=True)
               st.markdown(f'<p class="total-sum">{total_sum}</p>', unsafe_allow_html=True)
               
            # Afficher le nombre total de colonnes de new_row
            #  with col3:
            #     st.write("Number of Columns in New Row")
            #     st.write(len(new_row.columns))

            # Ajouter la nouvelle prédiction au DataFrame
             df_for_predictions = add_row(df_for_predictions, new_prediction)

            # Mettre à jour le graphique interactif
             #animate(df_for_predictions.tail(n_rows_to_display + 1), 'Note_Client', online_ts_chart)  # Ajouter 1 pour inclure la prédiction 

            # Afficher les étapes intermédiaires pour le débogage
            #  st.write("New Row:")
            #  st.write(new_row)

            # Créer un cadre avec un fond de couleur autour de la partie New Prediction
# Créer un cadre avec un fond de couleur autour de la partie New Prediction
             st.markdown("---")
             st.markdown("<div style='background-color: #f0f0f0; padding: 10px;'>", unsafe_allow_html=True)
             st.write("New Prediction:")
             st.write(new_prediction)
             st.markdown("</div>", unsafe_allow_html=True)
