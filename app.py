# import libraries for eda app in streamlit
import streamlit as st
import pandas as pd
import pandas_profiling
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# from pandafiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Make containers
header = st.container()
predication = st.container()
dataset = st.container()
eda = st.container()

# Model building and training
df = sns.load_dataset('iris')
X = df.drop('species', axis=1)
y = df.iloc[:, -1]
# Train model
model = LogisticRegression()
model.fit(X, y)

# function
def user_report():
    st.sidebar.title('Please set your parameters to predict the iris species')
    sepal_length = st.sidebar.slider('sepal_length', 4.0, 8.0, 0.2)
    sepal_width = st.sidebar.slider('sepal_width', 1.0, 4.5, 0.1)
    petal_length = st.sidebar.slider('petal_length', 1.0, 7.0, 0.5)
    petal_width = st.sidebar.slider('petal_width', 0.1, 3.0, 0.1)

    user_report_data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width,
        }
    return pd.DataFrame(user_report_data, index=[0])


user_data = user_report()


# model
if st.sidebar.button('Predict'):
    user_result = model.predict(user_data)
    with predication:
        st.subheader('Prediction')
        if user_result[0]=='setosa':
            st.write(user_result[0])
            st.markdown('![Setosa](https://i.pinimg.com/474x/5e/f0/ed/5ef0ed539518cb4486b70054152ef03a--form-irises.jpg)')
        elif user_result[0]=='versicolor':
            st.write(user_result[0])
            st.markdown('![Versicolor](https://th.bing.com/th/id/R.fab81b7ba812129560bec29c2a5a05f0?rik=fR3bSuRq4t9%2bEQ&riu=http%3a%2f%2fgreenvaluenursery.com%2fimage.php%3ftype%3dT%26id%3d18902&ehk=oMufiD4H5Y0HWMSvXCSVWMObTcDtJoazU14SQZyb08Y%3d&risl=&pid=ImgRaw&r=0)')
        elif user_result[0]=='virginica':
            st.write(user_result[0])
            st.markdown('![Virginica](https://th.bing.com/th/id/R.de768c27d0c619ac55d93d6ece03dc02?rik=iL9BUtB2aNQsnQ&riu=http%3a%2f%2fwww.missouriplants.com%2fimages%2fIris_virginica_inflorescence.jpg&ehk=vxNZ6tldyJkftE%2bSl1KICvum8CbHuDQBWSNrc%2f87g2A%3d&risl=&pid=ImgRaw&r=0)')
        else:
            st.write('No image')
else:
    with predication:
        st.subheader('Prediction')
        st.write('Click the button on sidebar to predict')

with header:
    st.title('Iris Flower Prediction App')
    st.subheader("Dataset exploration and iris species prediction")

with dataset:
    st.header('Iris Dataset')
    st.text('Iris Dataset')
    # show dataset
    st.write(df.head())
    st.markdown(' ***Do you want to see the EDA? click the button below***')
    if st.button('Show EDA'):
        with eda:
            pr = df.profile_report()
            st_profile_report(pr)

    


