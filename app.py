
from flask import Flask, render_template, request, send_from_directory, send_file
from forms import LoginForm
from lbxdscript import predict_rpart2, update_users
import os
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr



app = Flask(__name__)

app.secret_key = 'abcd'
update_users()
print('Users Updated')
rarray = [importr('utils'),importr('rpart'), importr('rattle'), importr('rpart.plot'),importr('stats'),importr('base'),importr('car')]
print('rarray')
@app.route("/", methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if request.method == 'POST':
        print(request.form)
        print(request.form['list'])
        array, rec = predict_rpart2(request.form['username'],request.form['list'], rarray)
        return render_template('recs.html', film1 = array[0], film2 = array[1], film3 = array[2], film4 = array[3], film5 = array[4], link1 = 'https://letterboxd.com/tmdb/' + str(rec[0]), link2 = 'https://letterboxd.com/tmdb/' + str(rec[1]), link3 = 'https://letterboxd.com/tmdb/' + str(rec[2]), link4='https://letterboxd.com/tmdb/' + str(rec[3]), link5='https://letterboxd.com/tmdb/' + str(rec[4]))
        
    return render_template("index.html", form = form)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8000)