from app import MultiApp
from PIL import Image
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import base64

app = MultiApp()

st.set_page_config(layout="wide")
def app0():
    st.markdown("""
    <style>
    .big-font {
        font-size:23px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("LIVER HEALTH DETECTION")
    st.write("By Abhishek Saigal, Class 12, The Shri Ram School, Aravali")
    st.title("INTRODUCTION")
    st.write("A healthy liver is a basic requirement for the overall health and wellness of an individual.")
    st.write("Presence of Liver Disease has a significant impact on the health of a person as the liver is responsible for over 500 functions in the body and is often referred to as the metabolic factory of our body.")
    st.write("It is therefore of immense use to have an app which is able to detect whether or not a person has a healthy liver. Furthermore, if there is an app which can predict the same without the need of a clinician at the initial assessment, it will be of great utility.")
    img = Image.open('Images/liver image 2.jpg')
    st.image(img, caption='A healthy Liver', width = 450)
    st.title("LIVER DISEASES")
    st.markdown('<p class="big-font">TYPES OF LIVER DISEASES:</p>', unsafe_allow_html=True)
    st.write("An unhealthy liver can be the sequel of several liver diseases stemming from various causes. Some of the common liver diseases are:")
    column_1, column_2 = st.beta_columns([3,5])
    with column_1:
        st.write("1. Acute Viral Hepatitis\n2. Chronic Viral Hepatitis\n3. Cirrhosis\n4. Alcoholic Liver Disease\n 5. Non-Alcoholic Fatty Liver Disease\n 6. Autoimmune Liver Disease\n7. Drug-induced Liver Disease\n8. Cholestatic Liver Disease\n9. Genetic and Metabolic Liver Diseases\n10. Liver Cancer\n")
        #st.write("These Diseases can be caused by a variety of factors such as:")
        #st.write("1. Hepatitis Viruses (A, B, C and E)\n2. Alcohol\n3. Matabolic causes\n4. Immune System Abnormalities\n5. Genetics\n6. Cancer\n7. Hepatotoxic Drugs")
    with column_2:
        st.image('Images/l2.jpg', use_column_width=True)

    st.markdown('<p class="big-font">CAUSES OF LIVER DISEASES:</p>', unsafe_allow_html=True)
    st.write("These Liver Diseases can be caused by a variety of factors such as:")
    column_3, column_4, column_5 = st.beta_columns([9,6,4])
    with column_3:
        st.write("1. Hepatitis Viruses (A, B, C and E)\n2. Alcohol\n3. Metabolic causes\n4. Immune System Abnormalities\n5. Genetic Abnormalities\n6. Cancer\n7. Hepatotoxic Drugs")
    with column_4:
        st.image('Images/Screen Shot 2021-05-16 at 11.11.48 PM.png', use_column_width=True)
    st.title("OBJECTIVES")
    st.write("1. To gain an understanding of Liver Diseases and the main factors causing them.\n2. To analyse the data and find out which features are most important in detecting Liver Disease.\n3. To develop a Liver Disease Prediction machine learning model which is able to perform at a relatively high accuracy (85-90%).")


def app1():
    st.title("THE DATASET")
    st.write("The Indian Liver Patient Records Dataset has patient records from North East Andhra Pradesh, India.")
    st.write("This dataset consists of 583 total records, with 416 liver patient records and 167 non-liver patient records.")
    st.write("The features present in this dataset were:")
    st.write("1. Age of the Patient\n2. Gender of the Patient\n3. Total Bilirubin\n4. Direct Bilirubin\n5. Alkaline Phosphotase\n6. Alanine Aminotransferase\n7. Aspartate Aminotransferase\n8. Total Proteins\n9. Albumin\n10. Albumin and Globulin Ratio")
    st.write("Each entry also has a label \"Dataset\" to split the data into persons with liver disease and those without liver disease.")
    liver = pd.read_csv('indian_liver_patient.csv')
    st.subheader('INDIAN LIVER PATIENTS DATASET:')
    st.dataframe(liver)
    st.subheader("KEY STATISTICS ABOUT THE DATASET")
    st.write(liver.describe())
    st.subheader("INITIAL DATA ANALYSIS: DISTRUBUTION PLOTS")
    #fig, ax = plt.subplots(figsize=(5, 2))
    #sns.distplot(liver['Aspartate_Aminotransferase'])
    #st.pyplot(fig)
    img = Image.open('Images/merge_from_ofoct (1).jpg')
    st.image(img, use_column_width = True)
    img = Image.open('Images/merge_from_ofoct (3).jpg')
    st.image(img, use_column_width = True)
    img = Image.open('Images/merge_from_ofoct (5).jpg')
    st.image(img, use_column_width = True)


def app2():
    liver = pd.read_csv('indian_liver_patient.csv')
    st.title('EXPLORATORY DATA ANALYSIS')
    st.title("Correlation Heatmap")
    st.write("By analysing a heatmap we can see the correlation between all the features in the dataset.")
    #cormap = liver.corr()
    #fig, ax = plt.subplots(figsize=(15,15))
    #sns.heatmap(cormap, cmap = "Greens", annot = True)
    #st.pyplot(fig)
    img = Image.open('Images/new_heatmap.png')
    st.image(img, use_column_width = True)

    st.title("Further Analysis:")
    #sns.set()
    #fig2 = sns.pairplot(liver, hue='Dataset', kind='reg')
    #st.pyplot(fig2)
    img = Image.open('Images/liver graphs.png')
    st.image(img, use_column_width = True)

    st.title("Highly Correlated Features")
    st.write("From these 2 plots, we have found 4 pairs of highly correlated features:")
    st.write("1. Direct Bilirubin and Total Bilirubin")
    st.write("2. Aspartate Aminotransferase and Alanine Aminotransferase")
    st.write("3. Total Proteins and Albumin")
    st.write("4. Albumin and Albumin and Globulin Ratio")
    st.title("Visualisation of correlated features:")
    #sns.set()
    #fig = sns.jointplot("Direct_Bilirubin", "Total_Bilirubin", data=liver, kind="reg")
    #st.pyplot(fig)
    img = Image.open('Images/corr_merge.jpg')
    st.image(img, width = 900)
    img = Image.open('Images/merge_from_ofoct (7).jpg')
    st.image(img, width = 900)

def app3():
    st.title("LIVER DISEASE PREDICTION")
    liver = pd.read_csv('indian_liver_patient.csv')
    liver.columns = liver.columns.map(str.lower)     
    liver.albumin_and_globulin_ratio.fillna(liver.albumin_and_globulin_ratio.mean(), inplace=True) 
    #liver.drop(['direct_bilirubin', 'aspartate_aminotransferase', 'albumin'], axis=1, inplace=True)
    skewed_cols = ['albumin_and_globulin_ratio','total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase']
    #for c in skewed_cols:
        #liver[c] = liver[c].apply('log1p')

    from sklearn.preprocessing import LabelEncoder, RobustScaler
    from sklearn.utils import resample
    from sklearn.model_selection import train_test_split

    label_enc = LabelEncoder()
    liver['gender'] = label_enc.fit_transform(liver['gender'])
    robust_sc = RobustScaler()
    #for c in liver[['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'albumin_and_globulin_ratio']].columns:
        #liver[c] = robust_sc.fit_transform(liver[c].values.reshape(-1, 1))
    liver.dataset.value_counts()
    minority = liver[liver.dataset==2]
    majority = liver[liver.dataset==1]
    minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])
    liver = pd.concat([minority_upsample, majority], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(liver.drop('dataset', axis=1), liver['dataset'], test_size=0.25, random_state=123)

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(random_state=123)
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    def get_input():
        age = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            age = st.slider('Age', 1, 90, 35)
        with col_2:
            bili = 0
            s_bili = st.text_input("Age", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                age = bili
        #st.write(age*2)
        input_gender = st.text_input("Gender", "")
        gender = 0
        if input_gender.lower() == "female":
            gender = 1

        total_bilirubin = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            total_bilirubin = st.slider('Total Bilirubin', 0.5, 15.0, 0.80)
        with col_2:
            bili = 0
            s_bili = st.text_input("Total Bilirubin", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                total_bilirubin = bili
        #st.write(total_bilirubin*2)
        direct_bilirubin = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            direct_bilirubin = st.slider('Direct Bilirubin', 0.3, 12.5, 0.30)
        with col_2:
            bili = 0
            s_bili = st.text_input("Direct Bilirubin", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                direct_bilirubin = bili
        #st.write(direct_bilirubin*2)
        alkaline_phosphotase = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            alkaline_phosphotase = st.slider('Alkaline Phosphotase', 50, 500, 111)
        with col_2:
            bili = 0
            s_bili = st.text_input("Alkaline Phosphotase", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                alkaline_phosphotase = bili
        #st.write(alkaline_phosphotase*2)
        alamine_amonotransferase = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            alamine_aminotransferase = st.slider('Alanine Aminotransferase', 20, 250, 31)
        with col_2:
            bili = 0
            s_bili = st.text_input("Alanine Aminotransferase", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                alamine_aminotransferase = bili
        #st.write(alamine_aminotransferase*2)
        aspartate_amonotransferase = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            aspartate_aminotransferase = st.slider('Aspartate Aminotransferase', 20, 500, 34)
        with col_2:
            bili = 0
            s_bili = st.text_input("Aspartate Aminotransferase", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                aspartate_aminotransferase = bili
        #st.write(aspartate_aminotransferase*2)
        total_proteins = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            total_proteins = st.slider('Total Proteins', 2.0, 10.0, 6.0)
        with col_2:
            bili = 0
            s_bili = st.text_input("Total Proteins", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                total_proteins = bili
        #st.write(total_proteins*2)
        albumin = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            albumin = st.slider('Albumin', 0.5, 5.0, 3.8)
        with col_2:
            bili = 0
            s_bili = st.text_input("Albumin", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                albumin = bili
        #st.write(albumin*2)
        albumin_and_globulin_ratio = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            albumin_and_globulin_ratio = st.slider('Albumin and Globulin Ratio', 0.1, 3.0, 1.73)
        with col_2:
            bili = 0
            s_bili = st.text_input("Albumin and Globulin Ratio", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                albumin_and_globulin_ratio = bili
        #st.write(albumin_and_globulin_ratio*2)
        user_data = {'age': age, 'gender':gender, 'total bilirubin':total_bilirubin, 'direct bilirubin':direct_bilirubin, 'alkaline phosphotase':alkaline_phosphotase, 'alanine aminotransferase':alamine_aminotransferase, 'aspartate aminotransferase':aspartate_aminotransferase, 'total proteins':total_proteins, 'albumin':albumin, 'albumin and globulin ratio':albumin_and_globulin_ratio}
    
        features = pd.DataFrame(user_data, index=[0])
        return features
    st.title('User Details:')
    st.write("Please enter the readings for the following values by moving the slider. In case your value does not fit in the given range, please enter it in the text box provided next to the slider. ")
    st.write("After entering your readings, please scroll down to the \"Result\" section to check whether or not you have a healthy Liver.")
    user_input = get_input()
    user_input.columns = user_input.columns.map(str.lower) 
    #print(user_input)
    #for c in user_input[['age', 'gender', 'total bilirubin', 'alkaline phosphotase', 'amaline aminotransferase', 'albumin and globulin ratio']].columns:
        #user_input[c] = robust_sc.fit_transform(user_input[c].values.reshape(-1, 1))
    #print(X_test.loc[[48]])
    #print(y_test_hat[48])
    st.dataframe(user_input)
    #user_input.drop('direct bilirubin', axis=1, inplace=True)
    #user_input.drop('aspartate aminotransferase', axis=1, inplace=True)
    #user_input.drop('total proteins', axis=1, inplace=True)
    #user_input.drop('albumin', axis=1, inplace=True)
    prediction = model.predict(user_input)
    st.title("Result:")
    if (prediction==1):
        st.write("You have Liver Disease.")
    else:
        st.write("You do not have Liver Disease.")
    
    st.title("Confusion Matrix")
    cm=confusion_matrix(y_test, y_test_hat)
    st.write(cm)
    #plot_confusion_matrix(model, y_test, y_test_hat)
    #st.pyplot()
    #img = Image.open('Images/confusion matrix.png')
    #st.image(img, width = 600)
    from sklearn import metrics
    st.title("Accuracy Score:")
    st.subheader(metrics.accuracy_score(y_test,y_test_hat)*100)
    
    import base64
    st.title('Visualisation of the Extra Trees Classifier')
    st.write("The Extremely Randomized Trees Classifier aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it’s classification result. Its classification visualisation can be seen below:")
    file_ = open("Images/randomforest.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="visualisation">',
        unsafe_allow_html=True,
    )

def app5():
    st.title("CONCLUSIONS")
    st.write("1. By using an appropriate methodology, standard Liver Function Tests (LFTs) can be used to predict whether a person has a healthy or a diseased liver with 85-90 percent accuracy.")
    st.write("2. The Extra Trees Classifier is an effective machine learning model for analysis of Medical Datasets such as Liver Function Tests.")

st.sidebar.title("PAGE NAVIGATION")
app.add_app("Home Page", app0)
app.add_app("The Dataset", app1)
app.add_app("Exploratory Data Analysis", app2)
app.add_app("Liver Disease Prediction", app3)
app.add_app("Conclusions", app5)
app.run()
