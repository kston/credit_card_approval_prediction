import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
import sklearn, threadpoolctl, scipy, imblearn
from lightgbm import LGBMClassifier
from datetime import date, datetime

from werkzeug.utils import redirect
app = Flask(__name__)

@app.route("/")
def index():
    modal_id = '#Amor'
    return render_template('index.html', modal_id=modal_id)


@app.route('/', methods=['POST'])
def submit_form():
 
    if request.method == 'POST':
        model = joblib.load("./ml/grid_model.pkl") 
        col_names = joblib.load("./ml/column_names.pkl")
        data = request.form.to_dict()
        # Convert JSON request to Pandas DataFrame
        df = pd.DataFrame([data])
        #transform total_month
        today = date.today()
        total_months = list(df['total_months'].values)
        date_become_clinte = datetime.strptime(total_months[0], '%Y-%m-%d').date()

        time_difference =  today - date_become_clinte
        df['total_months'] = round((time_difference.days/365)*12, 1)
    
        # Match Column Na,es
        df = df.reindex(columns=col_names)
        
        for col in ['flag_mobil', 'flag_work_phone','flag_phone', 'flag_email']:
            df[col] = [int(i) for i in df[col]]
        
        for column in df.columns:
            df[column] = df[column].astype(object)
        
        # Get prediction
        prediction = list(model.predict(df))
        # Return JSON version of Prediction
        print(prediction[0])
        if prediction[0] == 'Yes':
            modal_id = '#model_approved'
            return  render_template('index.html', modal_id= modal_id)
        elif prediction[0] == 'No':
            modal_id = '#modal_declained'
            return  render_template('index.html', modal_id=modal_id)

        #return jsonify({'prediction': str(prediction)})
    
    
    else:
        modal = "#modal_error"
        return  render_template('index.html', modal=modal)

@app.route("/card")
def card():
    return render_template('card.html')
