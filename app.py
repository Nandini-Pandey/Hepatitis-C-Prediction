import streamlit as st
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('HepatitisCdata.csv')

# Drop unnecessary columns
df.drop('Unnamed: 0', axis=1, inplace=True)

# Handling missing values
df['ALP'].replace(np.NaN, df['ALP'].mode()[0], inplace=True)
df['PROT'].replace(np.NaN, df['PROT'].mode()[0], inplace=True)
df['ALB'].replace(np.NaN, df['ALB'].mode()[0], inplace=True)
df['ALT'].replace(np.NaN, df['ALT'].mode()[0], inplace=True)
df['CHOL'].replace(np.NaN, df['CHOL'].mode()[0], inplace=True)

# Label encoding
df['Category'] = df['Category'].replace({
    '0=Blood Donor': 0, 
    '0s=suspect Blood Donor': 0, 
    '1=Hepatitis': 1, 
    '2=Fibrosis': 2, 
    '3=Cirrhosis': 3
})
df['Sex'] = df['Sex'].replace({'m': 0, 'f': 1})

# Handling class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(df.drop('Category', axis=1), df['Category'])
df_resampled = pd.DataFrame(X_resampled, columns=df.drop('Category', axis=1).columns)
df_resampled['Category'] = y_resampled

# Handling outliers
outlier_columns = ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
for column in outlier_columns:
    df_resampled[column] = np.log1p(df_resampled[column])

# Split the data into features (X) and target variable (Y)
X = df_resampled.drop('Category', axis=1)
Y = df_resampled['Category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Model selection
naive_bayes_classifier = GaussianNB()

# Model training
naive_bayes_classifier.fit(X_train, y_train)

# Streamlit web app interface
def main():
    st.title('Hepatitis C Diagnosis Predictor')

    # Collect user inputs
    st.sidebar.header('Enter Patient Details:')
    age = st.sidebar.text_input('Age', '0')
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    alb = st.sidebar.text_input('ALB (Albumin)', '0.0')
    alp = st.sidebar.text_input('ALP (Alkaline Phosphatase)', '0')
    alt = st.sidebar.text_input('ALT (Alanine Aminotransferase)', '0')
    ast = st.sidebar.text_input('AST (Aspartate Aminotransferase)', '0')
    bil = st.sidebar.text_input('BIL (Bilirubin)', '0.0')
    che = st.sidebar.text_input('CHE (Cholinesterase)', '0.0')
    chol = st.sidebar.text_input('CHOL (Cholesterol)', '0.0')
    crea = st.sidebar.text_input('CREA (Creatinine)', '0.0')
    ggt = st.sidebar.text_input('GGT (Gamma-Glutamyl Transferase)', '0')
    prot = st.sidebar.text_input('PROT (Protein)', '0.0')

    # Prepare user input data
    sex_encoded = 0 if sex == 'Male' else 1
    user_input = {
        'Age': float(age),
        'Sex': sex_encoded,
        'ALB': float(alb),
        'ALP': float(alp),
        'ALT': float(alt),
        'AST': float(ast),
        'BIL': float(bil),
        'CHE': float(che),
        'CHOL': float(chol),
        'CREA': float(crea),
        'GGT': float(ggt),
        'PROT': float(prot)
    }

    # Apply log transformation to user input
    for column in outlier_columns:
        user_input[column] = np.log1p(user_input[column])

    # Debugging: Display user input
    st.write("User Input:")
    st.write(pd.DataFrame([user_input]))

    # Predict button
    if st.button('Predict'):
        # Predict diagnosis category
        prediction = naive_bayes_classifier.predict(pd.DataFrame([user_input]))
        prediction_prob = naive_bayes_classifier.predict_proba(pd.DataFrame([user_input]))

        # Debugging: Display prediction probabilities
        st.write("Prediction Probabilities:")
        st.write(prediction_prob)

        # Display prediction result
        st.write('### Diagnosis Prediction:')
        
        category=prediction[0]
        if category==0:
            st.write(f'Predicted Category: Blood Donor/suspect Blood Donor')
        elif category==1:
           st.write(f'Predicted Category: Hepatitis')
        elif category==2:
            st.write(f'Predicted Category: Fibrosis')
        elif category==3:
            st.write(f'Predicted Category: Cirrhosis')

if __name__ == '__main__':
    main()
