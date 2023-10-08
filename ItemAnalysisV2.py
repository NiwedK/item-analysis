import streamlit as st
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import scipy
from sklearn.cluster import KMeans


def categorize_rank(rank):
    if rank <= upper_limit:
        return "Upper"
    elif rank >= lower_limit:
        return "Lower"
    else:
        return "Middle"

def KR_int(KR20_score):
    if KR20_score >= 0.90:
        return "Excellent reliability !!!; at the level of the best standardized test"
    elif KR20 >= 0.80:
        return "Very good for a classroom test"
    elif KR20 >= 0.70:
        return "Good for a classroom test; in the range of most. There are probably a few items which could be improved"
    elif KR20 >= 0.60:
        return "Somewhat low"
    elif KR20 >= 0.50:
        return "Suggests need for revision of test"
    else:
        return "Not recommended for testing"


st.set_page_config(page_title="Statistics", page_icon=":tada:", layout="wide")

# - - HEADER SECTION - -
with st.container():
    st.header("Item Analysis for MCQ with detailed analysis")
    st.write("This Item Analysis program was designed by Aj.Dr.Niwed Kullawong (2023), Health and Biomedical Analytics program, School of Health Science, MFU")

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
    cat_percentage = st.sidebar.selectbox("Select category % for group classification",[25,27,33.3,50], index=1)

    # Common task ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    df2 = df
    i, j = 0, 0
    for j in range(2, question_count + 2):
        for i in range(0, student_numbers):
            if df2.iloc[i + 1, j] == df2.iloc[0, j]:
                df2.iloc[i + 1, j] = 1
            else:
                df2.iloc[i + 1, j] = 0
            i = + 1
        j = + 1
    df2['Scores'] = df2.iloc[1:, 2:].sum(axis=1)
    df2['Rank'] = df2['Scores'].rank(method='min', ascending=False)

    n = np.ceil((cat_percentage/100) * student_numbers)
    N = np.ceil(2 * n)
    # st.write(df2)
    variance_T = df2['Scores'].var()
    st.sidebar.write('Total (T) variance is :', variance_T.round(1))
    st.sidebar.write("Expected count of students at " + str(cat_percentage) + " % (n) are:", n)
    st.sidebar.write("Expected count of students for analysis (N) are:", N)

    lower_limit = student_numbers - n
    upper_limit = n

    # Apply the categorize_rank function to create a new "Category" column
    df2['Category'] = df2['Rank'].apply(categorize_rank)
    df3['Scores'] = df2['Scores']
    df3['Rank'] = df2['Rank']
    df3['Category'] = df2['Category']
    df3.loc[0, 'Category'] = 'Reference'

    # remove "Middle" from df3
    df4 = df3[df3['Category'] != "Middle"]
    variance_N = df4['Scores'].var()

    df5 = df
    df6 = df5
    df5 = df5.astype(str)
    # st.write(df5)
    i, j = 0, 0
    for j in range(2, question_count + 2):
        for i in range(0, student_numbers):
            if df5.iloc[i + 1, j] == "1":
                df5.iloc[i + 1, j] = "Correct"
            elif df5.iloc[i + 1, j] == "1.0":
                df5.iloc[i + 1, j] = "Correct"
            elif df5.iloc[i + 1, j] == 1:
                df5.iloc[i + 1, j] = "Correct"
            else:
                df5.iloc[i + 1, j] = "Wrong"
            i = +1
        j = +1

    # Creating descriptive table
    #st.write(df3)
    descriptive_table = np.zeros([question_count, 13])
    descriptive_table = pd.DataFrame(descriptive_table,
                                     columns=['Question', 'N', 'It1', 'It2', 'It3', 'It4', 'C1', 'C2', 'C3', 'C4', 'Ans',
                                              'Correct', 'Wrong'])
    def format_float(val, decimal_places=2):
        return f'{val:.{decimal_places}f}'

    i = 0
    for i in range(0, len(question_list)):
        descriptive_table.iloc[i, 0] = question_list[i]
        descriptive_table.iloc[i, 1] = student_numbers
        value_counts = df3[question_list[i]].value_counts()
        value_unique = df3[question_list[i]].unique()
        ans = df3.loc[0, question_list[i]]
        condition_correct = (df3[question_list[i]] == ans)
        count_correct = df3.loc[condition_correct, question_list[i]].count()
        condition_wrong = (df3[question_list[i]] != ans)
        count_wrong = df3.loc[condition_wrong, question_list[i]].count()
        list_shape = len(value_unique)

        if list_shape == 4:
            item1 = value_unique[0]
            item2 = value_unique[1]
            item3 = value_unique[2]
            item4 = value_unique[3]
            descriptive_table.iloc[i, 2] = item1
            descriptive_table.iloc[i, 3] = item2
            descriptive_table.iloc[i, 4] = item3
            descriptive_table.iloc[i, 5] = item4
            descriptive_table.iloc[i, 6] = value_counts[item1]
            descriptive_table.iloc[i, 7] = value_counts[item2]
            descriptive_table.iloc[i, 8] = value_counts[item3]
            descriptive_table.iloc[i, 9] = value_counts[item4]
        elif list_shape == 3:
            item1 = value_unique[0]
            item2 = value_unique[1]
            item3 = value_unique[2]
            descriptive_table.iloc[i, 2] = item1
            descriptive_table.iloc[i, 3] = item2
            descriptive_table.iloc[i, 4] = item3
            descriptive_table.iloc[i, 5] = "-"
            descriptive_table.iloc[i, 6] = value_counts[item1]
            descriptive_table.iloc[i, 7] = value_counts[item2]
            descriptive_table.iloc[i, 8] = value_counts[item3]
            descriptive_table.iloc[i, 9] = 0
        elif list_shape == 2:
            item1 = value_unique[0]
            item2 = value_unique[1]
            descriptive_table.iloc[i, 2] = item1
            descriptive_table.iloc[i, 3] = item2
            descriptive_table.iloc[i, 4] = "-"
            descriptive_table.iloc[i, 5] = "-"
            descriptive_table.iloc[i, 6] = value_counts[item1]
            descriptive_table.iloc[i, 7] = value_counts[item2]
            descriptive_table.iloc[i, 8] = 0
            descriptive_table.iloc[i, 9] = 0
        else:
            item1 = value_unique[0]
            descriptive_table.iloc[i, 2] = item1
            descriptive_table.iloc[i, 3] = "-"
            descriptive_table.iloc[i, 4] = "-"
            descriptive_table.iloc[i, 5] = "-"
            descriptive_table.iloc[i, 6] = value_counts[item1]
            descriptive_table.iloc[i, 7] = 0
            descriptive_table.iloc[i, 8] = 0
            descriptive_table.iloc[i, 9] = 0

        descriptive_table.iloc[i, 10] = ans
        descriptive_table.iloc[i, 11] = (count_correct * 100 / student_numbers).round(2)
        descriptive_table.iloc[i, 12] = count_wrong * 100 / student_numbers
        i = +1

    descriptive_table['N'] = descriptive_table['N'].astype(int)
    descriptive_table['C1'] = descriptive_table['C1'].astype(int)
    descriptive_table['C2'] = descriptive_table['C2'].astype(int)
    descriptive_table['C3'] = descriptive_table['C3'].astype(int)
    descriptive_table['C4'] = descriptive_table['C4'].astype(int)
    descriptive_table['Correct'] = descriptive_table['Correct'].astype(float)
    descriptive_table['Wrong'] = descriptive_table['Wrong'].astype(float)
    descriptive_table['Correct'] = descriptive_table['Correct'].apply(lambda x: format_float(x, 1))
    descriptive_table['Wrong'] = descriptive_table['Wrong'].apply(lambda x: format_float(x, 1))

    #st.write(descriptive_table)

    # Creating reliability table
    #st.write(df4)
    report_table = np.zeros([question_count, 13])
    report_table = pd.DataFrame(report_table,
                                columns=['Question', 'N', 'WL', 'WU', 'CL', 'CU', 'Diff Index', 'Int-1', 'Disc Index',
                                         'Int-2', 'p', 'q', 'pq'])
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
        # st.table(cross_tab_df)
        cross_tab_df_buffer.update(cross_tab_df)
        cross_tab_df_buffer = cross_tab_df_buffer.astype(int)
        CU = cross_tab_df_buffer.iloc[0, 1]
        CL = cross_tab_df_buffer.iloc[0, 0]
        WL = cross_tab_df_buffer.iloc[1, 0]
        WU = cross_tab_df_buffer.iloc[1, 1]
        DI = (CU + CL) / analysis_student_count
        DisI = (CU - CL) / (analysis_student_count / 2)
        p = CU / analysis_student_count
        q = CL / analysis_student_count
        pq = p * q
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
    analysis_student_count = len(df4) - 1
    st.sidebar.write("Counts of students for analysis (N) are:", analysis_student_count)
    st.sidebar.write('Analysis (N) variance is :', variance_N.round(1))

    report_table2 = np.zeros([question_count, 8])
    report_table2 = pd.DataFrame(report_table2,
                                 columns=['Question', 'N', 'Correct', 'Wrong', 'p', 'q', 'pq', 'Collective pq'])
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
        report_table2.iloc[i, 4] = correct_count / student_numbers
        report_table2.iloc[i, 5] = wrong_count / student_numbers
        report_table2.iloc[i, 6] = ((correct_count / student_numbers) * (wrong_count / student_numbers))
        sum2_pq = sum2_pq + ((correct_count / student_numbers) * (wrong_count / student_numbers))
        report_table2.iloc[i, 7] = sum2_pq
        i = + 1
    report_table2['N'] = report_table2['N'].astype(int)
    report_table2['Correct'] = report_table2['Correct'].astype(int)
    report_table2['Wrong'] = report_table2['Wrong'].astype(int)

    work = st.sidebar.radio("Select process to work", (
        'Descriptive Analysis', 'Difficulty & Discrimination Analysis', 'Distractor Analysis', 'Reliability Analysis',
        'Question clustering and analysis', 'Student Grouping','Summary page'), index=0)

    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    if work == 'Descriptive Analysis':
        st.header("Descriptive Analysis")
        st.table(descriptive_table)

    if work == 'Difficulty & Discrimination Analysis':
        st.header("Difficulty & Discrimination Analysis")

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

    if work == 'Question clustering and analysis':
        st.header('Question clustering and analysis')
        selected_columns_df1 = descriptive_table[
            ['Question', 'N', 'It1', 'It2', 'It3', 'It4', 'C1', 'C2', 'C3', 'C4', 'Ans', 'Correct', 'Wrong']]
        selected_columns_df2 = report_table[
            ['WL', 'WU', 'CL', 'CU', 'Diff Index', 'Int-1', 'Disc Index', 'Int-2', 'p', 'q', 'pq']]
        question_analysis_df = pd.concat([selected_columns_df1, selected_columns_df2], axis=1)
        question_analysis_report_df = question_analysis_df[
            ['Question', 'WL', 'WU', 'CL', 'CU', 'Diff Index', 'Disc Index']]
        question_analysis_report_df = question_analysis_report_df.set_index('Question')

        # Choose the number of clusters (K)
        k = 3

        # Create a K-Means model
        kmeans = KMeans(n_clusters=k)

        # Fit the model to your data
        kmeans.fit(question_analysis_report_df)

        # Assign cluster labels to each data point
        labels = kmeans.predict(question_analysis_report_df)
        question_analysis_report_df['Q_Cluster'] = labels
        # st.write(question_analysis_report_df)
        question_cluster = question_analysis_report_df['Q_Cluster'].value_counts()
        # st.write(question_cluster)
        st.markdown("*Cluster map*")
        vmin = st.sidebar.slider("Select Range MIN", min_value=-2.0, max_value=-0.1, value=-1.0, step=0.1)
        vmax = st.sidebar.slider("Select Range MAX", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        font_size = st.sidebar.number_input("Input font size (pt)-The large number the small font size :", min_value=8,
                                            max_value=50, value=15, step=1)
        method_cluster = st.sidebar.selectbox("Select method for cluster",
                                              ['single', 'complete', 'average', 'weighted', 'centroid', 'median',
                                               'ward'], index=2)
        metric_cluster = st.sidebar.selectbox("Select metric",
                                              ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                                               'cosine', 'dice', 'euclidean',
                                               'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis',
                                               'matching', 'minkowski',
                                               'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                               'sokalsneath', 'sqeuclidean', 'yule'], index=7)

        fig3_clustermap = sns.clustermap(question_analysis_report_df, pivot_kws=None,
                                         method=method_cluster, metric=metric_cluster,
                                         z_score=1, standard_scale=None,
                                         vmin=vmin, vmax=vmax, center=0,
                                         annot=question_analysis_report_df, fmt='', cmap='RdBu', figsize=(font_size, font_size),
                                         row_cluster=True,
                                         col_cluster=False,
                                         row_colors=None,
                                         col_colors=None,
                                         mask=None,
                                         dendrogram_ratio=0.2,
                                         colors_ratio=0.03,
                                         cbar_pos=(1.00, 0.25, 0.03, 0.5),
                                         tree_kws=None)
        fig3_clustermap.fig.suptitle("Heatmap of Questions")
        # fig2_clustermap1.ax_heatmap.set_title('Subplot Title')
        st.pyplot(fig3_clustermap)

    if work == 'Student grouping':
        st.header('Student grouping')
        # st.write(df5)
        df5 = df5.drop(0)
        st.subheader('Student grouping based on scores')
        st.subheader("Upper group")
        upper_students = df5[df5['Category'] == 'Upper']
        upper_students = upper_students.sort_values(by='Scores', ascending=False)
        st.table(upper_students[['Student Name', 'Student ID', 'Scores', 'Category']])

        st.subheader("Middle group")
        middle_students = df5[df5['Category'] == 'Middle']
        middle_students = middle_students.sort_values(by='Scores', ascending=False)
        st.table(middle_students[['Student Name', 'Student ID', 'Scores', 'Category']])

        st.subheader("Lower group")
        lower_students = df5[df5['Category'] == 'Lower']
        lower_students = lower_students.sort_values(by='Scores', ascending=False)
        st.table(lower_students[['Student Name', 'Student ID', 'Scores', 'Category']])

        df6 = df6.drop(0)

        DDI_weight = report_table[['Diff Index', 'Disc Index']]
        DDI_weight['Weight'] = report_table['Diff Index']*report_table['Disc Index']

        i, j = 0, 0
        for j in range(2, question_count + 2):
            weight = DDI_weight.iloc[j-2,2]
            for i in range(0, student_numbers):
                df6.iloc[i, j] = df6.iloc[i, j]*weight
                i = +1
            j = +1
        n = question_count
        df6['Weight Scores'] = df6.iloc[0:,2:-3].sum(axis=1)
        df6['Weight Rank'] = df6['Weight Scores'].rank(method='max', ascending=False)
        df6['Weight Category'] = df6['Weight Rank'].apply(categorize_rank)

        weight_corr = df6[['Rank','Weight Rank']].corr()
        st.write(weight_corr)
        Weight_crosstab = pd.crosstab(index=df6['Category'], columns=df6['Weight Category'])
        st.write(Weight_crosstab)


    if work == 'Reliability Analysis':
        st.header('Reliability Analysis')
        with st.expander("Raw Calculation for reliability"):
            st.table(report_table2)

        col3, col4 = st.columns([2, 2])
        with col3:
            st.subheader("KR-20 Reliability test for Internal Consistency")
            # st.subheader("KR-20 Reliability test for Internal Consistency (N)")
            K = analysis_student_count
            sum_pq = report_table['pq'].sum()
            KR20 = (K / (K - 1)) * (1 - (sum_pq / variance_N))
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
            report_table3.iloc[0, 2] = KR_int(KR20)
            report_table3.iloc[1, 2] = KR_int(KR20_2)
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











    if work == 'Distractor Analysis':
        st.header('Distractor Analysis')
        #st.table(descriptive_table)
        df7 = df3
        df7 = df7[df7['Category'] != "Middle"]
        #st.write(df7)
        # Creating distractive table
        #st.write(df3)
        distractive_table = np.zeros([question_count, 15])
        distractive_table = pd.DataFrame(distractive_table,columns=['Question', 'N', 'It1', 'It2', 'It3', 'It4', '%1', '%2', '%3', '%4',
                                                  'Answer','It1-int','It2-int','It3-int','It4-int'])

        def format_float(val, decimal_places=2):
            return f'{val:.{decimal_places}f}'


        i = 0
        for i in range(0, len(question_list)):
            distractive_table.iloc[i, 0] = question_list[i]
            distractive_table.iloc[i, 1] = student_numbers
            value_counts_dis = df7[question_list[i]].value_counts()
            value_unique_dis = df7[question_list[i]].unique()
            #st.write(value_unique_dis)
            ans = df7.loc[0, question_list[i]]
            list_shape_dis = len(value_unique_dis)
            #st.write(list_shape_dis)

            if list_shape_dis == 4:
                item1_dis = value_unique_dis[0]
                item2_dis = value_unique_dis[1]
                item3_dis = value_unique_dis[2]
                item4_dis = value_unique_dis[3]
                distractive_table.iloc[i, 2] = item1_dis
                distractive_table.iloc[i, 3] = item2_dis
                distractive_table.iloc[i, 4] = item3_dis
                distractive_table.iloc[i, 5] = item4_dis
                distractive_table.iloc[i, 6] = value_counts_dis[item1_dis]*100/analysis_student_count
                distractive_table.iloc[i, 7] = value_counts_dis[item2_dis]*100/analysis_student_count
                distractive_table.iloc[i, 8] = value_counts_dis[item3_dis]*100/analysis_student_count
                distractive_table.iloc[i, 9] = value_counts_dis[item4_dis]*100/analysis_student_count

            elif list_shape_dis == 3:
                item1_dis = value_unique_dis[0]
                item2_dis = value_unique_dis[1]
                item3_dis = value_unique_dis[2]
                distractive_table.iloc[i, 2] = item1_dis
                distractive_table.iloc[i, 3] = item2_dis
                distractive_table.iloc[i, 4] = item3_dis
                distractive_table.iloc[i, 5] = "-"
                distractive_table.iloc[i, 6] = value_counts_dis[item1_dis]*100/analysis_student_count
                distractive_table.iloc[i, 7] = value_counts_dis[item2_dis]*100/analysis_student_count
                distractive_table.iloc[i, 8] = value_counts_dis[item3_dis]*100/analysis_student_count
                distractive_table.iloc[i, 9] = 0

            elif list_shape_dis == 2:
                item1_dis = value_unique_dis[0]
                item2_dis = value_unique_dis[1]
                distractive_table.iloc[i, 2] = item1_dis
                distractive_table.iloc[i, 3] = item2_dis
                distractive_table.iloc[i, 4] = "-"
                distractive_table.iloc[i, 5] = "-"
                distractive_table.iloc[i, 6] = value_counts_dis[item1_dis]*100/analysis_student_count
                distractive_table.iloc[i, 7] = value_counts_dis[item2_dis]*100/analysis_student_count
                distractive_table.iloc[i, 8] = 0
                distractive_table.iloc[i, 9] = 0

            else:
                item1_dis = value_unique_dis[0]
                distractive_table.iloc[i, 2] = item1_dis
                distractive_table.iloc[i, 3] = "-"
                distractive_table.iloc[i, 4] = "-"
                distractive_table.iloc[i, 5] = "-"
                distractive_table.iloc[i, 6] = value_counts_dis[item1_dis]*100/analysis_student_count
                distractive_table.iloc[i, 7] = 0
                distractive_table.iloc[i, 8] = 0
                distractive_table.iloc[i, 9] = 0
            distractive_table.iloc[i, 10] = ans
            if distractive_table.iloc[i, 6] > 5.00:
                distractive_table.iloc[i, 11] = "Good enough"
            else:
                distractive_table.iloc[i, 11] = "Not good enough"

            if distractive_table.iloc[i, 7] > 5.00:
                distractive_table.iloc[i, 12] = "Good enough"
            else:
                distractive_table.iloc[i, 12] = "Not good enough"
            if distractive_table.iloc[i, 8] > 5.00:
                distractive_table.iloc[i, 13] = "Good enough"
            else:
                distractive_table.iloc[i, 13] = "Not good enough"

            if distractive_table.iloc[i, 9] > 5.00:
                distractive_table.iloc[i, 14] = "Good enough"
            else:
                distractive_table.iloc[i, 14] = "Not good enough"
            i =+1
        distractive_table[['Diff Index','Disc Index']] = report_table[['Diff Index','Disc Index']]
        #st.table(distractive_table)

        distractive_report_table = np.zeros([4, 7])
        distractive_report_table = pd.DataFrame(distractive_report_table,
                                         columns=['Question', 'Item','Percent', 'Interpretation','Answer','Diff Index','Disc Index'])
        distractive_report_table2 = np.zeros([0, 7])
        distractive_report_table2 = pd.DataFrame(distractive_report_table2,
                                                columns=['Question', 'Item', 'Percent', 'Interpretation', 'Answer',
                                                         'Diff Index', 'Disc Index'])

        i = 0
        for i in range(0, len(question_list)):
            question_select = question_list[i]
            #st.write(question_select)
            distractive_table2 = distractive_table[distractive_table['Question'] == question_select]
            distractive_report_table.iloc[0, 0] = distractive_table2.iloc[0, 0]
            distractive_report_table.iloc[0, 1] = distractive_table2.iloc[0, 2]
            distractive_report_table.iloc[0, 2] = distractive_table2.iloc[0, 6]
            distractive_report_table.iloc[0, 3] = distractive_table2.iloc[0, 11]
            distractive_report_table.iloc[0, 4] = distractive_table2.iloc[0, 10]
            distractive_report_table.iloc[0, 5] = distractive_table2.iloc[0, 15]
            distractive_report_table.iloc[0, 6] = distractive_table2.iloc[0, 16]

            distractive_report_table.iloc[1, 0] = ""
            distractive_report_table.iloc[1, 1] = distractive_table2.iloc[0, 3]
            distractive_report_table.iloc[1, 2] = distractive_table2.iloc[0, 7]
            distractive_report_table.iloc[1, 3] = distractive_table2.iloc[0, 12]
            distractive_report_table.iloc[1, 4] = ""
            distractive_report_table.iloc[1, 5] = ""
            distractive_report_table.iloc[1, 6] = ""

            distractive_report_table.iloc[2, 0] = ""
            distractive_report_table.iloc[2, 1] = distractive_table2.iloc[0, 4]
            distractive_report_table.iloc[2, 2] = distractive_table2.iloc[0, 8]
            distractive_report_table.iloc[2, 3] = distractive_table2.iloc[0, 13]
            distractive_report_table.iloc[2, 4] = ""
            distractive_report_table.iloc[2, 5] = ""
            distractive_report_table.iloc[2, 6] = ""

            distractive_report_table.iloc[3, 0] = ""
            distractive_report_table.iloc[3, 1] = distractive_table2.iloc[0, 5]
            distractive_report_table.iloc[3, 2] = distractive_table2.iloc[0, 9]
            distractive_report_table.iloc[3, 3] = distractive_table2.iloc[0, 14]
            distractive_report_table.iloc[3, 4] = ""
            distractive_report_table.iloc[3, 5] = ""
            distractive_report_table.iloc[3, 6] = ""
            distractive_report_table2 = pd.concat([distractive_report_table2, distractive_report_table], ignore_index=True)
            i =+1
        distractive_report_table2 = distractive_report_table2.round(3)
        distractive_report_table2['Percent'] = distractive_report_table2['Percent'].apply(lambda x: format_float(x, 2))
        distractive_report_table2 = distractive_report_table2.set_index('Question')
        st.table(distractive_report_table2)
