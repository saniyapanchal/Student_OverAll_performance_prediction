import matplotlib
from sqlalchemy.dialects.mysql import pymysql
import mysql.connector as connector
matplotlib.use('Agg')
from flask import Flask, render_template, request, redirect, session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import dabl
import io
import base64
from math import *
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go


app = Flask(__name__)

def plot_to_img():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def get_student_performance_graph(data, column_name, title):
    counts = data[column_name].value_counts().reset_index()
    counts.columns = ['x', 'y']
    df = pd.DataFrame(counts)
    fig = px.pie(df, values='y', names='x', title=title)
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        width=500,
        height=600,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        legend_title_text='Marks',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    img = io.BytesIO()
    fig.write_image(img, format='png')
    img.seek(0)
    graph = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return graph



def get_overall_performance_graph(data):
    x = data['Class Test Marks'] + data['Assignment Marks'] + data['Attendance Marks'] + data['Mid Sem Marks'] + data['Final Exam']
    y = data['Student Name']
    fig, ax = plt.subplots()
    ax.bar(y, x)
    # Adjust the margins around the plot
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)
    plt.xticks(rotation=90, ha='center')
    fig.suptitle('Overall Performance', fontsize=20)
    plt.ylabel('Scored')
    plt.xlabel('Student Name')
    ax.margins(x=0.01)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return graph


def get_performance_percentage(data):
    x = data['Class Test Marks'] + data['Assignment Marks'] + data['Attendance Marks'] + data['Mid Sem Marks'] + data['Final Exam']
    max_marks = data.shape[0] * 100
    percentage = round((sum(x) / max_marks) * 100, 2)
    if percentage >= 70:
        remarks = "Good"
        color = "green"
    elif percentage >= 50 and percentage < 70:
        remarks = "Average"
        color = "orange"
    else:
        remarks = "Bad"
        color = "red"
    return f"{percentage}"

def get_performance_remarks(data):
    x = data['Class Test Marks'] + data['Assignment Marks'] + data['Attendance Marks'] + data['Mid Sem Marks'] + data['Final Exam']
    max_marks = data.shape[0] * 100
    percentage = round((sum(x) / max_marks) * 100, 2)
    if percentage >= 70:
        remarks = "Good"
        color = "green"
    elif percentage >= 50 and percentage < 70:
        remarks = "Average"
        color = "orange"
    else:
        remarks = "Bad"
        color = "red"
    return f"{remarks}"

app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')


# Function to create the database and table
def create_db():
    conn = connector.connect(host='localhost',
                                    user='root',
                                    password='',
                                    database='flask')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INT PRIMARY KEY AUTO_INCREMENT, username VARCHAR(255), password VARCHAR(255))''')
    c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", ('admin', '12345678'))
    conn.commit()
    conn.close()


# Initialize the database
create_db()

# Route to handle password update
@app.route('/update-password', methods=['POST'])
def update_password_route():
    # Get the username and new password from the form
    username = request.form['username']
    newpassword = request.form['newpassword']

    # Update the password for the given username
    if update_password(username, newpassword):
        return redirect('/index.html')
    else:
        return 'Failed to update password'

# Function to update password
def update_password(username, newpassword):
    conn = connector.connect(host='localhost', user='root', password='', database='flask')
    c = conn.cursor()
    c.execute("UPDATE users SET password = %s WHERE username = %s", (newpassword, username))
    if c.rowcount > 0:
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

# Route to handle password update
@app.route('/forgot-password', methods=['POST'])
def forgot_password_route():
    # Get the current password and new password from the form
    username = request.form['username']
    password = request.form['password']
    newpassword = request.form['newpassword']

    # Update the password if the current password is correct
    if forgot_password(username, password, newpassword):
        return redirect('/upload.html')
    else:
        return 'Incorrect password'

# Function to update password
def forgot_password(username, password, newpassword):
    conn = connector.connect(host='localhost', user='root', password='', database='flask')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = %s", (username,))
    row = c.fetchone()
    if row and row[0] == password:
        c.execute("UPDATE users SET password = %s WHERE username = %s", (newpassword, username))
        conn.commit()
        conn.close()
        return True
    conn.close()

# @app.route('/upload.html', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username == 'admin' and password == '12345678':
#             return redirect('/upload.html')
#         else:
#             return 'Invalid username or password'
#     return render_template('upload.html')

@app.route('/upload.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = connector.connect(host='localhost', user='root', password='', database='flask')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = %s", (username,))
        row = c.fetchone()
        if row and row[0] == password:
            return redirect('/upload.html')
        else:
            return 'Invalid username or password'
        conn.close()
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        data = pd.read_csv(file)

        # Attendance Marks
        attendance_graph = get_student_performance_graph(data, 'Attendance Marks', 'Attendance Marks Distribution')

        # Class Test Marks
        classtest_graph = get_student_performance_graph(data, 'Class Test Marks', 'Class Test Marks Distribution')

        # Assignment Marks
        assignment_graph = get_student_performance_graph(data, 'Assignment Marks', 'Assignment Marks Distribution')

        # Mid Sem Marks
        midsem_graph = get_student_performance_graph(data, 'Mid Sem Marks', 'Mid Sem Marks Distribution')

        # Final Exam Marks
        finalexam_graph = get_student_performance_graph(data, 'Final Exam', 'Final Exam Marks Distribution')

        # Overall Performance Graph
        overall_performance_graph = get_overall_performance_graph(data)

        # Performance Percentage
        performance_percentage = get_performance_percentage(data)

        # Performance Remarks
        performance_remarks = get_performance_remarks(data)

        # Weakness Analysis
        # Class Test Marks Distribution Graphs
        sns.histplot(data=data, x='Class Test Marks', bins=10)
        plt.title('Distribution of Class Test Marks')
        plt.xlabel('Class Test Marks (out of 10)')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        class_test = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Assignment Marks Distribution Graphs
        sns.histplot(data=data, x='Assignment Marks', bins=10)
        plt.title('Distribution of Assignment Marks')
        plt.xlabel('Assignment Marks (out of 10)')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        assignment = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Attendance Marks Distribution Graphs
        sns.histplot(data=data, x='Attendance Marks', bins=10)
        plt.title('Distribution of Attendance')
        plt.xlabel('Attendance')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        attendance = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Mid Sem Marks Distribution Graphs
        sns.histplot(data=data, x='Mid Sem Marks', bins=10)
        plt.title('Distribution of Mid Sem Marks')
        plt.xlabel('Mid Sem Marks (out of 30)')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        mid_sem = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Final Exam Marks Distribution Graphs
        sns.histplot(data=data, x='Final Exam', bins=10)
        plt.title('Distribution of Final Exam Marks')
        plt.xlabel('Final Exam Marks (out of 100)')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        final_exam = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Calculate the average marks for each assessment type
        class_test_avg = data['Class Test Marks'].mean()
        assignment_avg = data['Assignment Marks'].mean()
        attendance_avg = data['Attendance Marks'].mean()
        mid_sem_avg = data['Mid Sem Marks'].mean()
        final_exam_avg = data['Final Exam'].mean()
        # Create a dictionary of assessment types and their average marks
        avg_marks = {'Class Test': class_test_avg, 'Assignment': assignment_avg, 'Attendance': attendance_avg, 'Mid Sem': mid_sem_avg, 'Final Exam': final_exam_avg}
        # Convert the dictionary to a pandas DataFrame
        df_avg_marks = pd.DataFrame.from_dict(avg_marks, orient='index', columns=['Average Marks'])
        # Reset the index to get the assessment types as a column
        df_avg_marks = df_avg_marks.reset_index().rename(columns={'index': 'Assessment Type'})
        # Create a bar plot of the average marks for each assessment type
        fig1 = px.bar(df_avg_marks, x='Assessment Type', y='Average Marks', title='Average Marks for Each Assessment Type')
        # Calculate the correlation matrix for the data
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = data[numerical_columns].corr()
        # Create a heatmap of the correlation matrix
        fig2 = px.imshow(corr_matrix, color_continuous_scale='blues', labels=dict(x='Assessment Type', y='Assessment Type', color='Correlation'))

        # Handle missing values
        data.dropna(inplace=True)
        # Convert data types
        data['Student Name'] = data['Student Name'].astype(str)
        data['Class Test'] = data['Class Test Marks'].astype(float)
        data['Assignment'] = data['Assignment Marks'].astype(float)
        data['Attendance'] = data['Attendance Marks'].astype(float)
        data['Mid Sem'] = data['Mid Sem Marks'].astype(float)
        data['Final Exam'] = data['Final Exam'].astype(float)
        # Create new feature
        data['Total Marks'] = data['Class Test'] + data['Assignment'] + data['Attendance'] + data['Mid Sem'] + data['Final Exam']
        from sklearn.model_selection import train_test_split
        X = data.drop(['Student Name', 'Final Exam'], axis=1)
        y = data['Final Exam']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        #print('Mean Squared Error:', mse)
        #print('R2 Score:', r2)
        data['Predicted Final Exam'] = model.predict(X)
        def performance_level(marks):
            if marks >= 130:
                return 'Excellent'
            elif marks >= 100:
                return 'Good'
            elif marks >= 90:
                return 'Average'
            elif marks >= 60:
                return 'Below Average'
            else:
                return 'Poor'
        data['Performance'] = data['Total Marks'].apply(performance_level)

        # Create scatter plot
        fig = px.scatter(data, y='Student Name', x='Total Marks', size='Final Exam', color='Final Exam', hover_data=['Student Name', 'Assignment Marks', 'Attendance Marks', 'Mid Sem Marks', 'Predicted Final Exam'])
        fig.update_layout(
            xaxis_title='Total Marks',
            yaxis_title='Student Name',
            font=dict(family='Arial', size=12),
            height=1200,
            width=1000,
            hovermode='closest'
        )

        # Generate plot div
        plot_div = fig.to_html(full_html=False)


        # Sending back the data to the server to the results.html file for output.
        return render_template('results.html', attendance_graph=attendance_graph, assignment_graph=assignment_graph, midsem_graph=midsem_graph, classtest_graph=classtest_graph, finalexam_graph=finalexam_graph, overall_performance_graph=overall_performance_graph, performance_percentage=performance_percentage, performance_remarks=performance_remarks, class_test=class_test, assignment=assignment, attendance=attendance, mid_sem=mid_sem, final_exam=final_exam, fig1=fig1, fig2=fig2, plot_div=plot_div, data=data)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
