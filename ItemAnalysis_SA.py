import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def categorize_rank(rank):
    if rank <= upper_limit:
        return "Upper"
    elif rank >= lower_limit:
        return "Lower"
    else:
        return "Middle"

def kr_int(KR20_score):
    if KR20_score >= 0.90:
        return "Excellent reliability !!!; at the level of the best standardized test"
    elif KR20_score >= 0.80:
        return "Very good for a classroom test"
    elif KR20_score >= 0.70:
        return "Good for a classroom test; in the range of most. There are probably a few items which could be improved"
    elif KR20_score >= 0.60:
        return "Somewhat low"
    elif KR20_score >= 0.50:
        return "Suggests need for revision of test"
    else:
        return "Not recommended for testing"


st.set_page_config(page_title="HSH-Item Analysis-Essay V1.0", page_icon=":tada:", layout="wide")
# - - HEADER SECTION - -

with st.container():
    st.header("Item Analysis for Essay question V1.0")
    st.write("This program was designed by Aj.Dr.Niwed Kullawong, HBA program, SHS, MFU, 2024")

col01, col02 = st.columns([2, 3])

with col01:
    st.header("Input a data file (.xlsx)")
    uploaded_file = st.file_uploader("Pick a file", type=".xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.fillna(0)

    # Define new column names for the first two columns
    new_column_names = ['Student Name', 'Student ID']

    # Change the column names of the first two columns
    df.columns = new_column_names + list(df.columns[2:])
    df3 = pd.read_excel(uploaded_file)
    df3 = df3.fillna("Missing")
    df3.columns = new_column_names + list(df3.columns[2:])
    question_count = len(df.columns[2:])
    question_list = df.columns[2:]
    columns = df.columns
    student_numbers = len(df[1:])

    with col02:
        st.write("Numbers of questions are :", question_count)
        st.write("Numbers of student responses are (T):", student_numbers)

    df2 = df
    df2['Scores'] = df2.iloc[0:, 2:].sum(axis=1)
    df2['Rank'] = df2['Scores'].rank(method='min', ascending=False)
    n = np.ceil(0.27*student_numbers)
    N = np.ceil(2*n)
    variance_T = df2['Scores'].var()
    #st.table(df2)

    with col02:
        st.write('Total (T) variance is :', variance_T.round(1))
        st.write("Expected count of students at 27% (n) are:", n)
        st.write("Expected count of students for analysis (N) are:", N)
    lower_limit = student_numbers - n
    upper_limit = n

    # Apply the categorize_rank function to create a new "Category" column
    df2['Category'] = df2['Rank'].apply(categorize_rank)
    df3['Scores'] = df2['Scores']
    df3['Rank'] = df2['Rank']
    df3['Category'] = df2['Category']

    #st.table(df3)
    #variance_N = df4['Scores'].var()
    report_table = np.zeros([question_count, 6])
    report_table = pd.DataFrame(report_table, columns=['Question', 'N', 'Agavrage Score', 'Full mark',  'Diff Index', 'Int-1'])

    #st.table(report_table)

    #df_mean = df3.iloc[1:, :].mean()
    #st.write(df_mean)
    i = 0
    for i in range(0, len(question_list)):
        report_table.iloc[i, 0] = question_list[i]
        report_table.iloc[i, 1] = student_numbers
        report_table.iloc[i, 2] = df3[question_list[i]].iloc[1:].mean()
        report_table.iloc[i, 3] = df3[question_list[i]].iloc[0]
        report_table.iloc[i, 4] = df3[question_list[i]].iloc[1:].mean()/df3[question_list[i]].iloc[0]
        if report_table.iloc[i, 4] >= 0.8:
            report_table.iloc[i, 5] = "Easy -> Modify"
        elif report_table.iloc[i, 4] >= 0.3:
            report_table.iloc[i, 5] = "Moderate -> Accept"
        else:
            report_table.iloc[i, 5] = "Easy -> Modify"
        i += 1

    with st.expander("Summary Table (Click to check)"):
        st.table(report_table.round(3))

    st.header("Summary of the test")
    #cross_tab2 = report_table.plot.bar(x='Int-1')
    dif = pd.DataFrame(report_table['Int-1'])
    dif_cat_count = report_table['Int-1'].value_counts()

    st.subheader("Visualization")
    cola, colb = st.columns([2, 4])
    with cola:
        fig01, ax = plt.subplots(figsize=(4, 4))
        plt.bar(dif_cat_count.index, dif_cat_count.values)
        #plt.title('Difficulty x Discrimination')
        plt.xlabel('Difficulty')
        plt.ylabel('Counts')
        st.write(fig01)
