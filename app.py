from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import sklearn
from sklearn.svm import LinearSVC



# load the model from disk
scaler = pickle.load(open('scaler.pkl','rb'))
clf = pickle.load(open('simple_cancer_pred.pkl', 'rb'))


app = Flask(__name__)


def fl_lst(text_1):

    return [float(x) for x in text_1.split(',')]


@app.errorhandler(500)
def page_not_found(e):
    
    return render_template('err_handle.html'), 500
        

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['message']
		data = fl_lst(message)
		my_prediction = int(clf.predict(scaler.transform([data])))
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
