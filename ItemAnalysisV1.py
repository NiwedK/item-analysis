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


st.set_page_config(page_title="Statistics", page_icon=":tada:", layout="wide")
# - - HEADER SECTION - -
with st.container():
    st.header("Item Analysis for MCQ")
    st.write("This program is designed by Aj.Dr.Niwed Kullawong, HBA program, SHS, MFU")
st.sidebar.header("User Input Data (.xlsx)")
uploaded_file = st.sidebar.file_uploader("Pick a file", type=".xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # Define new column names for the first two columns
    new_column_names = ['Student Name', 'Student ID']
    # Change the column names of the first two columns
    df.columns = new_column_names + list(df.columns[2:])
    df3 = pd.read_excel(uploaded_file)
    df3.columns = new_column_names + list(df3.columns[2:])
    question_count = len(df.columns[2:])
    question_list = df.columns[2:]
    columns = df.columns
    student_numbers = len(df[1:])
    st.sidebar.write("Numbers of questions are :", question_count)
    st.sidebar.write("Numbers of student response are (T):", student_numbers)

    df2 = df
    i, j = 0, 0
    for j in range(2, question_count+2):
        for i in range(0, student_numbers):
            if df2.iloc[i+1, j] == df2.iloc[0, j]:
                df2.iloc[i + 1, j] = 1
            else:
                df2.iloc[i + 1, j] = 0
            i = + 1
        j = + 1
    df2['Scores'] = df2.iloc[1:, 2:].sum(axis=1)
    df2['Rank'] = df2['Scores'].rank(method='min', ascending=False)
    n = np.ceil(0.27*student_numbers)
    N = np.ceil(2*n)
# st.write(df2)
    variance_T = df2['Scores'].var()
    st.sidebar.write('Total (T) variance is :', variance_T.round(1))
    st.sidebar.write("Expected count of students at 27% (n) are:", n)
    st.sidebar.write("Expected count of students for analysis (N) are:", N)
    lower_limit = student_numbers - n
    upper_limit = n
    # Apply the categorize_rank function to create a new "Category" column
    df2['Category'] = df2['Rank'].apply(categorize_rank)
    df3['Scores'] = df2['Scores']
    df3['Rank'] = df2['Rank']
    df3['Category'] = df2['Category']
    df3.loc[0, 'Category'] = 'Reference'
    df4 = df3[df3['Category'] != "Middle"]
    variance_N = df4['Scores'].var()
    report_table = np.zeros([question_count, 13])
    report_table = pd.DataFrame(report_table, columns=['Question', 'N', 'WL', 'WU', 'CL', 'CU', 'Diff Index', 'Int-1', 'Disc Index', 'Int-2', 'p', 'q', 'pq'])
    report_table['N'] = report_table['N'].astype(int)
    report_table['WL'] = report_table['WL'].astype(int)
    report_table['WU'] = report_table['WU'].astype(int)
    report_table['CL'] = report_table['CL'].astype(int)
    report_table['CU'] = report_table['CU'].astype(int)

    i = 0
    for i in range(0, len(question_list)):
        descriptive_var = ["Student Name", "Student ID", question_list[i], "Category"]
        df_descriptive = df4[descriptive_var]
        # st.write(df_descriptive)

        def check_result(row):
            if row[question_list[i]] == df_descriptive.iloc[0][question_list[i]]:
                return 'Correct'
            else:
                return 'Wrong'


        df_descriptive['Result'] = df_descriptive.apply(check_result, axis=1)
        analysis_student_count = len(df_descriptive) - 1
        cross_tab = pd.crosstab(df_descriptive.iloc[1:]['Result'], df_descriptive.iloc[1:]['Category'])
        cross_tab_df = pd.DataFrame(cross_tab)
        cross_tab_df_buffer = np.zeros([2, 2])
        cross_tab_df_buffer = pd.DataFrame(cross_tab_df_buffer, index=['Correct', 'Wrong'], columns=['Lower', 'Upper'])
        cross_tab_df_buffer.update(cross_tab_df)
        cross_tab_df_buffer = cross_tab_df_buffer.astype(int)
        CU = cross_tab_df_buffer.iloc[0, 1]
        CL = cross_tab_df_buffer.iloc[0, 0]
        WL = cross_tab_df_buffer.iloc[1, 0]
        WU = cross_tab_df_buffer.iloc[1, 1]
        DI = (CU + CL) / analysis_student_count
        DisI = (CU - CL) / (analysis_student_count / 2)
        p = CU/analysis_student_count
        q = CL/analysis_student_count
        pq = p*q
        report_table.iloc[i, 0] = question_list[i]
        report_table.iloc[i, 1] = analysis_student_count
        report_table.iloc[i, 2] = WL
        report_table.iloc[i, 3] = CL
        report_table.iloc[i, 4] = WU
        report_table.iloc[i, 5] = CU
        report_table.iloc[i, 6] = DI.round(3)
        if DI >= 0.76:
            report_table.iloc[i, 7] = "Easy -> Revise/Discard"
        elif DI >= 0.26:
            report_table.iloc[i, 7] = "Right Difficulty -> Retain"
        else:
            report_table.iloc[i, 7] = "High Difficulty -> Revise/Discard"
        report_table.iloc[i, 8] = DisI.round(3)
        if DisI >= 0.500:
            report_table.iloc[i, 9] = "Very Good Item -> Very Usable"
        elif DisI >= 0.400:
            report_table.iloc[i, 9] = "Good Item -> Very Usable"
        elif DisI >= 0.300:
            report_table.iloc[i, 9] = "Fair Quality -> Usable"
        elif DisI >= 0.200:
            report_table.iloc[i, 9] = "Potential Poor Item -> Consider Revising"
        else:
            report_table.iloc[i, 9] = "Very Poor Item -> Consider Revising/Discard"
        report_table.iloc[i, 10] = p
        report_table.iloc[i, 11] = q
        report_table.iloc[i, 12] = pq
        i += 1
    with st.expander("Summary Table (Click to check)"):
        st.table(report_table.round(3))

    st.header("Summary of the test")
    cross_tab2 = pd.crosstab(report_table['Int-1'], report_table['Int-2'])
    st.table(cross_tab2)

    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Visualization")
        fig1, ax = plt.subplots(figsize=(8, 8))
        cross_tab2.plot(kind='bar', stacked=True, ax=ax)
        plt.title('Difficulty x Discrimination')
        plt.xlabel('Difficulty')
        plt.ylabel('Counts')
        st.write(fig1)
        # st.pyplot(barplot)

    analysis_student_count = len(df4)-1
    st.sidebar.write("Counts of students for analysis (N) are:", analysis_student_count)
    st.sidebar.write('Analysis (N) variance is :', variance_N.round(1))

    report_table2 = np.zeros([question_count, 8])
    report_table2 = pd.DataFrame(report_table2, columns=['Question', 'N', 'Correct', 'Wrong', 'p', 'q', 'pq', 'Collective pq'])

    df5 = df
    df5 = df5.astype(str)
    # st.write(df5)
    i, j = 0, 0
    for j in range(2, question_count+2):
        for i in range(0, student_numbers):
            if df5.iloc[i+1, j] == "1":
                df5.iloc[i+1, j] = "Correct"
            elif df5.iloc[i+1, j] == "1.0":
                df5.iloc[i+1, j] = "Correct"
            elif df5.iloc[i+1, j] == 1:
                df5.iloc[i+1, j] = "Correct"
            else:
                df5.iloc[i+1, j] = "Wrong"
            i = +1
        j = +1

    i = 0
    sum2_pq = 0
    for i in range(0, len(question_list)):
        report_table2.iloc[i, 0] = question_list[i]
        report_table2.iloc[i, 1] = student_numbers
        result_count = df5[question_list[i]].value_counts()
        wrong_count = result_count.get("Wrong", 0)
        correct_count = result_count.get("Correct", 0)
        report_table2.iloc[i, 2] = correct_count
        report_table2.iloc[i, 3] = wrong_count
        report_table2.iloc[i, 4] = correct_count/student_numbers
        report_table2.iloc[i, 5] = wrong_count/student_numbers
        report_table2.iloc[i, 6] = ((correct_count/student_numbers)*(wrong_count/student_numbers))
        sum2_pq = sum2_pq + ((correct_count/student_numbers)*(wrong_count/student_numbers))
        report_table2.iloc[i, 7] = sum2_pq
        i = + 1
    report_table2['N'] = report_table2['N'].astype(int)
    report_table2['Correct'] = report_table2['Correct'].astype(int)
    report_table2['Wrong'] = report_table2['Wrong'].astype(int)

    with st.expander("Raw Calculation for reliability"):
        st.table(report_table2)

    # col3, col4 = st.columns([2, 2])
    with col2:
        st.subheader("KR-20 Reliability test for Internal Consistency")
        # st.subheader("KR-20 Reliability test for Internal Consistency (N)")
        K = analysis_student_count
        sum_pq = report_table['pq'].sum()
        KR20 = (K/(K-1))*(1-(sum_pq/variance_N))
        # st.subheader("KR-20 Reliability test for Internal Consistency (T)")
        K2 = student_numbers
        sum_pq2 = report_table2['pq'].sum()
        KR20_2 = (K2 / (K2 - 1)) * (1 - (sum_pq2 / variance_T))
        report_table3 = np.zeros([2, 3])
        report_table3 = pd.DataFrame(report_table3, columns=['Sample count', 'KR-20', 'Interpretation'])

        report_table3.iloc[0, 0] = analysis_student_count
        report_table3.iloc[1, 0] = student_numbers
        report_table3.iloc[0, 1] = KR20.round(2)
        report_table3.iloc[1, 1] = KR20_2.round(2)
        report_table3.iloc[0, 2] = kr_int(KR20)
        report_table3.iloc[1, 2] = kr_int(KR20_2)
        report_table3['Sample count'] = report_table3['Sample count'].astype(int)
        report_table3 = report_table3.set_index('Sample count')

        # st.subheader("Reliability result")
        fig2, ax = plt.subplots(figsize=(8, 8))
        report_table3.plot(kind='bar', stacked=True, ax=ax)
        plt.title('Reliability Scores')
        plt.xlabel('Sample')
        plt.ylabel('Score')
        st.write(fig2)
        st.table(report_table3)
