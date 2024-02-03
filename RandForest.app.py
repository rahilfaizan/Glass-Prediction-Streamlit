import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
# Glass Type Prediction 
""")

st.image('glassimg.jpg')
# fetch dataset 
glass = pd.read_csv("glass.csv")


st.sidebar.header('User Input Parameters')

def user_input_features():
    RI = st.sidebar.slider('Refractive Index', 1.511,1.540,1.511,0.005)
    Na = st.sidebar.slider('Sodium (Na)', 10.0, 18.0, 10.0, 0.5)
    Mg = st.sidebar.slider('Magnesium (Mg)', 0.0, 5.0, 0.0, 0.5)
    Al = st.sidebar.slider('Aluminum (Al)', 0.0, 4.0, 0.0, 0.5)
    Si = st.sidebar.slider('Silicon (Si)', 69.0, 76.0, 69.0, 0.5)
    K = st.sidebar.slider('Potassium (K)', 0.0, 3.0, 0.0, 0.25)
    Ca = st.sidebar.slider('Calcium (Ca)', 5.0, 17.0, 5.0, 0.5)
    Ba = st.sidebar.slider('Barium (Ba)', 0.0, 4.0, 0.0, 0.25)
    Fe = st.sidebar.slider('Iron (Fe)', 0.0, 0.6, 0.0, 0.05)


    data = {'RI': RI,
            'Na': Na,
            'Mg': Mg,
            'Al': Al,
            'Si': Si,
            'K': K,
            'Ca': Ca,
            'Ba': Ba,
            'Fe': Fe}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

X = glass.iloc[:,:9]
Y=glass["Type_of_glass"]
clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# Feature Importance
feature_importance = clf.feature_importances_
feature_names = X.columns
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Plotting
st.subheader('Feature Importance')
fig, ax = plt.subplots()
df_importance.sort_values(by='Importance', ascending=False, inplace=True)
ax.barh(df_importance['Feature'], df_importance['Importance'])
st.pyplot(fig)

# Plotting the predicted class probabilities
st.subheader('Prediction Probability')

# Convert prediction_proba to a DataFrame for better visualization
proba_df = pd.DataFrame(prediction_proba, columns=[f'{i}' for i in clf.classes_])

# Plotting
fig, ax = plt.subplots()
sns.barplot(x=proba_df.columns, y=proba_df.iloc[0], ax=ax)
ax.set_ylabel('Probability')
ax.set_title('Class Probabilities')
ax.set_ylim([0, 1])

st.pyplot(fig)
