import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from fpdf import FPDF

# Installation required: pip install factor-analyzer
from factor_analyzer import FactorAnalyzer


def interpret_omega(omega_val):
    if omega_val >= 0.80:
        color = "green"
        msg = "Excellent: very high reliability"
    elif omega_val >= 0.70:
        color = "blue"
        msg = "Good: standard reliability"
    else:
        color = "red"
        msg = "Warning: low reliability, please check questions with low factor loading"

    return msg, color


def calculate_omega(data_frame):
    """
    data_frame: Student score data (0/1) (Test columns only)
    """
    # 1. Check data readiness (Exclude constant columns)
    df_clean = data_frame.loc[:, (data_frame != data_frame.iloc[0]).any()]

    # 2. Exploratory Factor Analysis (EFA) 1 Factor
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(df_clean)

    # 3. Get Factor Loadings (λ)
    loadings = fa.loadings_.flatten()

    # 4. Calculate Unique Variances (ψ)
    # ψ = 1 - λ²
    uniquenesses = fa.get_uniquenesses()

    # 5. Omega Formula
    sum_loadings_sq = np.sum(loadings) ** 2
    sum_uniquenesses = np.sum(uniquenesses)

    omega_value = sum_loadings_sq / (sum_loadings_sq + sum_uniquenesses)

    return round(omega_value, 3)


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


st.set_page_config(page_title="SHS-MCQ Item Analysis V1.3", page_icon=":tada:", layout="wide")
# - - HEADER SECTION - -

with st.container():
    st.header("Item Analysis for MCQ V1.3")
    st.write("This program is designed by Asst.Prof.Dr.Niwed Kullawong, HBA program, SHS, MFU")

col01, col02 = st.columns([2, 3])

with col01:
    st.subheader("Input a data file (.xlsx)")
    uploaded_file = st.file_uploader("Pick a file", type=".xlsx")

# --- ส่วนการคำนวณคะแนนและจัดกลุ่มนักศึกษา ---

if uploaded_file:
    # 1. โหลดข้อมูล (สมมติว่าแถวที่ 0 คือเฉลย)
    df = pd.read_excel(uploaded_file).fillna(0)

    # กำหนดชื่อคอลัมน์ให้ชัดเจน (คอลัมน์ 0=ID, 1=Name, 2 เป็นต้นไป=ข้อสอบ)
    df.columns = ['Student ID', 'Student Name'] + list(df.columns[2:])

    question_list = df.columns[2:]  # รายชื่อข้อสอบเริ่มที่ Index 2
    question_count = len(question_list)
    student_responses = df.iloc[1:].copy()  # แยกเฉพาะแถวคำตอบนักศึกษา (ไม่เอาแถวเฉลย)
    answer_key = df.iloc[0, 2:]  # ดึงเฉลยจากแถวแรก เริ่มที่คอลัมน์ที่ 2

    # 2. ตรวจให้คะแนน (Scoring)
    # สร้าง DataFrame ใหม่สำหรับเก็บคะแนน 0/1
    scores_df = student_responses.copy()
    for q in question_list:
        key = answer_key[q]
        # เทียบคำตอบนักศึกษากับเฉลย
        scores_df[q] = scores_df[q].apply(lambda x: 1 if x == key else 0)

    # รวมคะแนนดิบ (เริ่มบวกจากคอลัมน์ที่ 2 เป็นต้นไป)
    scores_df['Total_Score'] = scores_df.iloc[:, 2:].sum(axis=1)

    # 3. จัดกลุ่ม Upper / Lower (27% Technique)
    num_students = len(scores_df)
    n_27 = int(np.ceil(0.27 * num_students))

    # เรียงลำดับคะแนนจากมากไปน้อย
    scores_df = scores_df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)


    def assign_group(index, total_n, n27):
        if index < n27:
            return "Upper"
        elif index >= (total_n - n27):
            return "Lower"
        else:
            return "Middle"


    # สร้างคอลัมน์ Category เพื่อระบุกลุ่ม
    scores_df['Category'] = [assign_group(i, num_students, n_27) for i in range(num_students)]

    # แสดงผลสถิติเบื้องต้นใน Streamlit
    st.success(f"Analysis for {num_students} students (High group/low group each {n_27} persons)")

    # แสดงตารางคะแนนและการจัดกลุ่ม (5 แถวแรก)
    with st.expander("Click for Student grouping"):
        st.write("Student grouping (Sorted by Score):")
        st.dataframe(scores_df[['Student ID', 'Student Name', 'Total_Score', 'Category']])

    # --- ส่วนนี้คุณสามารถนำไปใช้คำนวณ Difficulty (p) และ Discrimination (r) ต่อได้เลย ---
    # โดยใช้ค่าจาก scores_df[scores_df['Category'] == 'Upper'] และ 'Lower'

    new_column_names = ['Student Name', 'Student ID']
    df.columns = new_column_names + list(df.columns[2:])

    df3 = pd.read_excel(uploaded_file)
    df3 = df3.fillna("Missing")
    df3.columns = new_column_names + list(df3.columns[2:])

    question_count = len(df.columns[2:])
    question_list = df.columns[2:]
    columns = df.columns
    student_numbers = len(df[1:])

    with col02:
        st.write("Number of questions:", question_count)
        st.write("Number of student responses (T):", student_numbers)

    # Scoring process
    df2 = df.copy()
    for j in range(2, question_count + 2):
        for i in range(0, student_numbers):
            if df2.iloc[i + 1, j] == df2.iloc[0, j]:
                df2.iloc[i + 1, j] = 1
            else:
                df2.iloc[i + 1, j] = 0

    df2['Scores'] = df2.iloc[1:, 2:].sum(axis=1)
    df2['Rank'] = df2['Scores'].rank(method='min', ascending=False)
    n = np.ceil(0.27 * student_numbers)
    N = np.ceil(2 * n)

    variance_T = df2['Scores'].var()
    with col02:
        st.write('Total (T) variance:', variance_T.round(1))
        st.write("Expected count of students at 27% (n):", n)
        st.write("Expected count of students for analysis (N):", N)

    lower_limit = student_numbers - n
    upper_limit = n

    df2['Category'] = df2['Rank'].apply(categorize_rank)
    df3['Scores'] = df2['Scores']
    df3['Rank'] = df2['Rank']
    df3['Category'] = df2['Category']
    df3.loc[0, 'Category'] = 'Reference'

    df4 = df3[df3['Category'] != "Middle"]
    variance_N = df4['Scores'].var()

    # Create an empty list to store row data, then convert to DataFrame at the end
    # OR initialize with object type to allow strings and numbers
    report_table = pd.DataFrame(index=range(question_count), 
                            columns=['Question', 'N', 'WL', 'WU', 'CL', 'CU', 'Diff Index', 'Int-1', 'Disc Index', 'Int-2', 'p', 'q', 'pq'])

    # Ensure numeric columns are initialized if necessary, but keep the whole DF flexible
    report_table = report_table.astype(object)

    report_table['N'] = report_table['N'].astype(int)
    report_table['WL'] = report_table['WL'].astype(int)
    report_table['WU'] = report_table['WU'].astype(int)
    report_table['CL'] = report_table['CL'].astype(int)
    report_table['CU'] = report_table['CU'].astype(int)

    # Item Analysis Loop
    for i in range(0, len(question_list)):
        q_name = question_list[i]
        descriptive_var = ["Student Name", "Student ID", q_name, "Category"]
        df_descriptive = df4[descriptive_var].copy()


        def check_result(row):
            if row[q_name] == df_descriptive.iloc[0][q_name]:
                return 'Correct'
            else:
                return 'Wrong'


        df_descriptive['Result'] = df_descriptive.apply(check_result, axis=1)
        analysis_student_count = len(df_descriptive) - 1

        cross_tab = pd.crosstab(df_descriptive.iloc[1:]['Result'], df_descriptive.iloc[1:]['Category'])
        cross_tab_df_buffer = pd.DataFrame(0, index=['Correct', 'Wrong'], columns=['Lower', 'Upper'])
        cross_tab_df_buffer.update(cross_tab)

        CU = cross_tab_df_buffer.loc['Correct', 'Upper']
        CL = cross_tab_df_buffer.loc['Correct', 'Lower']
        WL = cross_tab_df_buffer.loc['Wrong', 'Lower']
        WU = cross_tab_df_buffer.loc['Wrong', 'Upper']

        DI = (CU + CL) / analysis_student_count
        DisI = (CU - CL) / (analysis_student_count / 2)
        p = CU / analysis_student_count
        q = CL / analysis_student_count  # Note: Usually q = 1-p, but keeping original logic
        pq = p * q

        report_table.iloc[i, 0] = q_name
        report_table.iloc[i, 1] = analysis_student_count
        report_table.iloc[i, 2] = WL
        report_table.iloc[i, 3] = CL
        report_table.iloc[i, 4] = WU
        report_table.iloc[i, 5] = CU
        report_table.iloc[i, 6] = DI.round(3)

        # Difficulty Interpretation
        if DI >= 0.76:
            report_table.iloc[i, 7] = "Easy -> Revise/Discard"
        elif DI >= 0.26:
            report_table.iloc[i, 7] = "Right Difficulty -> Retain"
        else:
            report_table.iloc[i, 7] = "High Difficulty -> Revise/Discard"

        # Discrimination Interpretation
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

    with st.expander("Summary Table (Click to check)"):
        st.table(report_table.round(3))

    st.header("Summary of the Test")
    cross_tab2 = pd.crosstab(report_table['Int-1'], report_table['Int-2'])

    # Visualization
    st.subheader("Interactive Visualization: Difficulty x Discrimination")
    diff_order = ["Easy -> Revise/Discard", "Right Difficulty -> Retain", "High Difficulty -> Revise/Discard"]
    disc_order = [
        "Very Poor Item -> Consider Revising/Discard",
        "Potential Poor Item -> Consider Revising",
        "Fair Quality -> Usable",
        "Good Item -> Very Usable",
        "Very Good Item -> Very Usable"
    ]

    df_plot = cross_tab2.reset_index().melt(id_vars='Int-1', var_name='Int-2', value_name='Count')
    df_plot['Int-1'] = pd.Categorical(df_plot['Int-1'], categories=diff_order, ordered=True)
    df_plot['Int-2'] = pd.Categorical(df_plot['Int-2'], categories=disc_order, ordered=True)
    df_plot = df_plot.sort_values(['Int-1', 'Int-2'])

    fig1 = px.bar(
        df_plot,
        x='Int-1',
        y='Count',
        color='Int-2',
        title='Item Quality Analysis Matrix (Sorted)',
        labels={'Int-1': 'Difficulty Level', 'Count': 'Number of Items', 'Int-2': 'Discrimination'},
        barmode='stack',
        color_discrete_map={
            "Very Poor Item -> Consider Revising/Discard": "#ff4d4d",
            "Potential Poor Item -> Consider Revising": "#ffaf40",
            "Fair Quality -> Usable": "#fffa65",
            "Good Item -> Very Usable": "#32ff7e",
            "Very Good Item -> Very Usable": "#3ae374"
        },
        category_orders={"Int-1": diff_order, "Int-2": disc_order}
    )
    fig1.update_layout(xaxis_title="Difficulty Level", yaxis_title="Total Question Counts",
                       legend_title="Discrimination Quality", height=600)
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap
    st.subheader("Heatmap Analysis: Test Quality Matrix (Sorted)")
    sorted_cross_tab = cross_tab2.reindex(index=diff_order, columns=disc_order, fill_value=0)
    fig_heat = go.Figure(data=go.Heatmap(
        z=sorted_cross_tab.values,
        x=sorted_cross_tab.columns,
        y=sorted_cross_tab.index,
        colorscale='YlGnBu',
        text=sorted_cross_tab.values,
        texttemplate="%{text}",
        textfont={"size": 16, "family": "Arial"},
        hoverinfo='x+y+z',
        showscale=True
    ))
    fig_heat.update_layout(title='Distribution of Test Items (Sorted Matrix)',
                           xaxis_title="Discrimination Level (Low → High)",
                           yaxis_title="Difficulty Level (Easy → High)", yaxis={'autorange': 'reversed'}, height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    analysis_student_count = len(df4) - 1
    with col02:
        st.write("Counts of students for analysis (N):", analysis_student_count)
        st.write('Analysis (N) variance:', variance_N.round(1))

    # Reliability Preparation
    report_table2 = pd.DataFrame(np.zeros([question_count, 8]),
                                 columns=['Question', 'N', 'Correct', 'Wrong', 'p', 'q', 'pq', 'Collective pq'])

    df5 = df.astype(str)
    for j in range(2, question_count + 2):
        for i in range(0, student_numbers):
            val = df5.iloc[i + 1, j]
            if val in ["1", "1.0"]:
                df5.iloc[i + 1, j] = "Correct"
            else:
                df5.iloc[i + 1, j] = "Wrong"

    sum2_pq = 0
    for i in range(0, len(question_list)):
        q_name = question_list[i]
        report_table2.iloc[i, 0] = q_name
        report_table2.iloc[i, 1] = student_numbers
        result_count = df5[q_name].value_counts()
        wrong_count = result_count.get("Wrong", 0)
        correct_count = result_count.get("Correct", 0)
        p_val = correct_count / student_numbers
        q_val = wrong_count / student_numbers
        report_table2.iloc[i, 2] = correct_count
        report_table2.iloc[i, 3] = wrong_count
        report_table2.iloc[i, 4] = p_val
        report_table2.iloc[i, 5] = q_val
        report_table2.iloc[i, 6] = p_val * q_val
        sum2_pq += (p_val * q_val)
        report_table2.iloc[i, 7] = sum2_pq

    with st.expander("Raw Calculation for Reliability"):
        st.table(report_table2)

    # Reliability Gauges
    st.subheader("Reliability Analysis (Comparison: KR-20 vs Coefficient Omega)")
    data_for_omega = df2.iloc[1:, 2:question_count + 2].astype(float)

    K_val = student_numbers
    sum_pq_val = report_table2['pq'].sum()
    KR20_final = (K_val / (K_val - 1)) * (1 - (sum_pq_val / variance_T)) if (K_val > 1 and variance_T > 0) else 0

    try:
        omega_final = calculate_omega(data_for_omega)
    except:
        omega_final = 0


    def draw_gauge_chart(value, title, color_bar="#2c3e50"):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value, title={'text': title, 'font': {'size': 18}},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color_bar},
                   'steps': [{'range': [0, 0.6], 'color': "#ff7675"},
                             {'range': [0.6, 0.8], 'color': "#ffe66d"},
                             {'range': [0.8, 1.0], 'color': "#55efc4"}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'value': 0.7}}))
        fig.update_layout(height=320)
        return fig


    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(draw_gauge_chart(KR20_final, "KR-20 (Traditional)"), use_container_width=False)
        st.info(f"**Interpretation (KR-20):** {kr_int(KR20_final)}")
    with col_right:
        st.plotly_chart(draw_gauge_chart(omega_final, "Omega (Modern)", color_bar="#0984e3"), use_container_width=False)
        msg_om, _ = interpret_omega(omega_final)
        st.success(f"**Interpretation (ω):** {msg_om}")

    st.write("### Summary Detail")
    st.table(pd.DataFrame({
        'Metric': ['KR-20 Reliability', 'Coefficient Omega (ω)'],
        'Score': [round(KR20_final, 3), round(omega_final, 3)],
        'Sample (N)': [student_numbers, student_numbers]
    }).set_index('Metric'))

    good_items_count = df_plot[(df_plot['Int-1'] == "Right Difficulty -> Retain") & (
        df_plot['Int-2'].isin(["Good Item -> Very Usable", "Very Good Item -> Very Usable"]))]['Count'].sum()
    st.success(f"🎯 Excellent Quality Items (Best Items) found: {int(good_items_count)} out of {question_count} items")

    # Distractor Analysis
    st.subheader('Distractor Analysis')
    df7 = df4.copy()
    distractive_table = pd.DataFrame(np.zeros([question_count, 15]),
                                     columns=['Question', 'N', 'It1', 'It2', 'It3', 'It4', '%1', '%2', '%3', '%4',
                                              'Answer', 'It1-int', 'It2-int', 'It3-int', 'It4-int'])

    for i in range(0, len(question_list)):
        q_name = question_list[i]
        distractive_table.iloc[i, 0] = q_name
        distractive_table.iloc[i, 1] = student_numbers
        v_counts = df7[q_name].value_counts()
        v_uniques = df7[q_name].unique()
        ans = df7.loc[0, q_name]

        for idx in range(4):
            if idx < len(v_uniques):
                item_val = v_uniques[idx]
                pct = v_counts[item_val] * 100 / analysis_student_count
                distractive_table.iloc[i, 2 + idx] = item_val
                distractive_table.iloc[i, 6 + idx] = pct
                distractive_table.iloc[i, 11 + idx] = "Good enough" if pct > 5.0 else "Not good enough"
            else:
                distractive_table.iloc[i, 2 + idx] = "-"
                distractive_table.iloc[i, 6 + idx] = 0
                distractive_table.iloc[i, 11 + idx] = "Not good enough"
        distractive_table.iloc[i, 10] = ans

    distractive_report_table2 = pd.DataFrame(
        columns=['Question', 'Item', 'Percent', 'Interpretation', 'Answer', 'Diff Index', 'Disc Index'])
    for i in range(0, len(question_list)):
        q_name = question_list[i]
        temp_row = distractive_table[distractive_table['Question'] == q_name]
        for idx in range(4):
            distractive_report_table2 = pd.concat([distractive_report_table2, pd.DataFrame([{
                'Question': q_name if idx == 0 else "",
                'Item': temp_row.iloc[0, 2 + idx],
                'Percent': f"{temp_row.iloc[0, 6 + idx]:.2f}",
                'Interpretation': temp_row.iloc[0, 11 + idx],
                'Answer': temp_row.iloc[0, 10] if idx == 0 else "",
                'Diff Index': report_table.iloc[i, 6] if idx == 0 else "",
                'Disc Index': report_table.iloc[i, 8] if idx == 0 else ""
            }])], ignore_index=True)

    st.table(distractive_report_table2.set_index('Question'))

    # Recommendations
    st.subheader("🎯 Test Improvement Recommendations")


    def get_recommendation(p, r):
        if r < 0:
            return "🔴 Discard: Defective item (Lower group scored better than upper)", "error"
        elif r < 0.20:
            return "🔴 Discard/Major Revise: Very poor discrimination", "error"
        elif r < 0.30:
            return "🟡 Revise: Fair discrimination, improve distractors", "warning"
        elif p < 0.20 or p > 0.80:
            return "🟡 Revise: Good discrimination but extreme difficulty", "warning"
        else:
            return "🟢 Retain: High quality item", "success"


    res_list = []
    for _, row in report_table.iterrows():
        rec, status = get_recommendation(row['p'], row['Disc Index'])
        res_list.append({'Question': row['Question'], 'Recommendation': rec, 'Status': status})
    res_df = pd.DataFrame(res_list)

    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Keep (Retain)", len(res_df[res_df['Status'] == 'success']))
    cs2.metric("Improve (Revise)", len(res_df[res_df['Status'] == 'warning']))
    cs3.metric("Remove (Discard)", len(res_df[res_df['Status'] == 'error']))

    with st.expander("Detailed Recommendations per Item"):
        for _, r in res_df.iterrows():
            if r['Status'] == "success":
                st.success(f"**{r['Question']}**: {r['Recommendation']}")
            elif r['Status'] == "warning":
                st.warning(f"**{r['Question']}**: {r['Recommendation']}")
            else:
                st.error(f"**{r['Question']}**: {r['Recommendation']}")

    st.plotly_chart(px.pie(res_df, names='Status', title='Overall Quality Distribution', color='Status',
                           color_discrete_map={'success': '#55efc4', 'warning': '#ffe66d', 'error': '#ff7675'}),
                    use_container_width=True)

    # Mis-key Detection
    st.subheader("🚨 Mis-key Detection")
    st.write("This section provides observations for review; the final decision rests with the test designer.")
    suspects = report_table[(report_table['Disc Index'] < 0) | (report_table['p'] < 0.15)]
    if not suspects.empty:
        st.warning("Potential mis-keyed items detected:")
        mk_list = []
        for _, row in suspects.iterrows():
            q_n = row['Question']
            u_resp = df3[(df3['Category'] == 'Upper') & (df3['Category'] != 'Reference')][q_n]
            if not u_resp.empty:
                counts = u_resp.value_counts()
                if not counts.empty:
                    top_choice = counts.idxmax()
                    curr_key = df3.loc[0, q_n]
                    if top_choice != curr_key:
                        mk_list.append({"Question": q_n, "Current Key": curr_key, "Top Upper Choice": top_choice,
                                        "Disc (r)": row['Disc Index'], "Diff (p)": row['p'],
                                        "Suggestion": f"Check if key should be '{top_choice}'"})
        if mk_list:
            st.table(pd.DataFrame(mk_list))
        else:
            st.info(
                "Low statistics found, but the Upper group still primarily chooses the current key (possibly very hard items).")
    else:
        st.success("✅ No mis-keyed patterns detected.")

    # Export
    st.subheader("📥 Export Analysis Report")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        report_table.to_excel(writer, sheet_name='Item_Analysis', index=False)
        distractive_report_table2.to_excel(writer, sheet_name='Distractor_Analysis', index=False)
    st.download_button(label="Download Full Report (Excel)", data=buffer, file_name="Item_Analysis_Report.xlsx",
                       mime="application/vnd.ms-excel")


    # --- PDF Report Generator Section (No Kaleido Required) ---

    class ItemAnalysisPDF(FPDF):
        def header(self):
            # Arial standard font
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Psychometric Analysis Report', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, 'Generated by SHS-MCQ Item Analysis System', 0, 1, 'C')
            # Horizontal line
            self.line(10, 30, 200, 30)
            self.ln(10)

        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


    def generate_english_pdf(report_table, omega_val, kr20_val):
        pdf = ItemAnalysisPDF()
        pdf.add_page()

        # --- Section 1: Reliability Summary ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "1. Overall Test Reliability", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"- McDonald's Omega (w): {omega_val:.3f}", ln=True)
        pdf.cell(0, 8, f"- Kuder-Richardson 20 (KR-20): {kr20_val:.3f}", ln=True)
        pdf.ln(5)

        # --- Section 2: Note about Charts ---
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "(Note: Interactive charts are available on the digital dashboard only)", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # --- Section 3: Item Diagnostic Table ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "2. Item-by-Item Diagnostic", ln=True)

        # Table Header
        pdf.set_fill_color(230, 230, 230)
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(20, 8, "Item", 1, 0, 'C', True)
        pdf.cell(20, 8, "Diff (p)", 1, 0, 'C', True)
        pdf.cell(20, 8, "Disc (r)", 1, 0, 'C', True)
        pdf.cell(130, 8, "Automated Recommendation", 1, 1, 'C', True)

        # Table Content
        pdf.set_font("Arial", '', 7)
        for _, row in report_table.iterrows():
            # Get Recommendation Text
            rec_text, _ = get_recommendation(row['p'], row['Disc Index'])
            # Strip Emojis for PDF compatibility
            clean_rec = rec_text.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")

            # Check for page break
            if pdf.get_y() > 270:
                pdf.add_page()
                # Repeat Header
                pdf.set_fill_color(230, 230, 230)
                pdf.set_font("Arial", 'B', 8)
                pdf.cell(20, 8, "Item", 1, 0, 'C', True)
                pdf.cell(20, 8, "Diff (p)", 1, 0, 'C', True)
                pdf.cell(20, 8, "Disc (r)", 1, 0, 'C', True)
                pdf.cell(130, 8, "Automated Recommendation", 1, 1, 'C', True)
                pdf.set_font("Arial", '', 7)

            pdf.cell(20, 7, str(row['Question']), 1, 0, 'C')
            pdf.cell(20, 7, f"{row['p']:.3f}", 1, 0, 'C')
            pdf.cell(20, 7, f"{row['Disc Index']:.3f}", 1, 0, 'C')
            pdf.cell(130, 7, clean_rec, 1, 1, 'L')

        return pdf.output(dest='S').encode('latin-1')


    # --- Streamlit UI Integration (Add this at the end of your script) ---
    st.write("---")
    st.subheader("📥 Export Final Report")
    if st.button("Generate PDF Summary"):
        try:
            # Generate the PDF bytes
            pdf_bytes = generate_english_pdf(report_table, omega_final, KR20_final)

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="Psychometric_Analysis_Report.pdf",
                mime="application/pdf"
            )
            st.success("PDF generated successfully!")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
