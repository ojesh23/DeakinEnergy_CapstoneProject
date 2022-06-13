"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
try:
    ################## Import required libraries ##################
    import os
    from flask import Flask, render_template, request
    import pickle
    from csv import writer
    from classifiers import RandomForest, SGD, SVC, RNN, LOGISTICREGRESSION, VotingClassifier, XgBoost
except ImportError:
    print("Required libraries are not installed properly, try again! ")
################## Start Flask App ##################
app = Flask(__name__, static_url_path='', static_folder='static')
app.config['USE_FAVICON'] = os.path.exists(os.path.join(app.static_folder, "favicon.ico"))
# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app
# global variables
empty_prediction_probabilities = {'rf_prep': '', 'rf_prob': '', 'sgd_pred': '', 'sgd_prob': '', 'svc_pred': '',
                                  'svc_prob': '', 'rnn_pred': '', 'rnn_prob': '', 'lr_pred': '', 'lr_prob': '',
                                  'voting_pred': '', 'voting_prob': '', 'xgb_pred':'', 'xgb_prob':''
                                  }

################## Base page ##################
@app.route("/analysis")
def base():
    return render_template('base.html')


################## Pandas Report ##################
@app.route("/pandas")
def pandas():
    return render_template('pandas_profile_report.html')

################## dtale ##################
@app.route("/dtale_3D_scatter")
def dtale():
    return render_template('dtale_3D_scatter.html')


############# Landing Page #############
@app.route("/")
def landingPage():
    return render_template('landingPage.html')

################## ML classifiers ##################
@app.route("/classifier", methods=['GET'])
def homepage():
    prediction_probabilities = {'rf_prep': '', 'rf_prob': '', 'sgd_pred': '', 'sgd_prob': '', 'svc_pred': '',
                                'svc_prob': '', 'rnn_pred': '', 'rnn_prob': '', 'lr_pred': '', 'lr_prob': '',
                                'voting_pred': '', "voting_prob": '', 'xgb_pred':'', 'xgb_prob':''}
    return render_template('index.html', prediction_probabilities=empty_prediction_probabilities)

################## Page not found ##################
@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

try:
    ######################### Classifier/predictions ############################
    @app.route('/predict', methods=['POST'])
    @app.route('/predictall', methods=['POST'])
    def prediction():
        def get_inputs():
            """
            get user inputs from form
            """
            classifier_name = str(request.form.get('classifier'))
            incident_description = str(request.form.get('event'))
            incident_causes = str(request.form.get('cause'))
            incident_consequences = str(request.form.get('consequence'))
            # concatenate all fields
            incident = incident_description + ' ' + incident_causes + ' ' + incident_consequences
            return classifier_name, incident, incident_description, incident_causes, incident_consequences
        # make prediction for all classifier
        def predict_all():
            classifier_name, _, incident_description, incident_causes, incident_consequences = get_inputs()
            prediction_probabilities = models("all")
            return render_template('index.html', scroll='table', incident=incident_description, cause=incident_causes,
                                   consequence=incident_consequences, prediction_probabilities=prediction_probabilities)
        # prediction for single classifier onlygit
        def make_prediction():
            classifier_name, _, incident_description, incident_causes, incident_consequences = get_inputs()
            # if no classifier is selected, display all classifiers prediction
            if classifier_name in ['select_classifier']:
                classifier_name = 'all'
            prediction_probabilities = models(classifier_name)
            return render_template('index.html', scroll='table', incident=incident_description, cause=incident_causes,
                                   consequence=incident_consequences, prediction_probabilities=prediction_probabilities)
        def models(classifier_name):
            # reset
            prediction_probabilities = {}
            prediction_probabilities.fromkeys(empty_prediction_probabilities, "")
            _, _, incident_description, _, _ = get_inputs()
            if classifier_name in ("all", "rf"):
                random_forest = RandomForest(incident_description)
                pred, prob, _ = random_forest.rf_prediction()
                prediction_probabilities['rf_pred'] = pred
                prediction_probabilities['rf_prob'] = prob
            if classifier_name in ("all", "sgd"):
                sgd = SGD(incident_description)
                pred, prob, _ = sgd.sgd_prediction()
                prediction_probabilities['sgd_pred'] = pred
                prediction_probabilities['sgd_prob'] = prob
            if classifier_name in ("all", "svc"):
                svc = SVC(incident_description)
                pred, prob, _ = svc.svc_prediction()
                prediction_probabilities['svc_pred'] = pred
                prediction_probabilities['svc_prob'] = prob
            if classifier_name in ("all", "rnn"):
                rnn = RNN(incident_description)
                pred, prob, _ = rnn.rnn_prediction()
                prediction_probabilities['rnn_pred'] = pred
                prediction_probabilities['rnn_prob'] = prob
            if classifier_name in ("all", "lr"):
                lr = LOGISTICREGRESSION(incident_description)
                pred, prob, _ = lr.lr_prediction()
                prediction_probabilities['lr_pred'] = pred
                prediction_probabilities['lr_prob'] = prob
            if classifier_name in ("all", "xgb"):
                xgb = XgBoost(incident_description)
                pred, prob, _ = xgb.xgb_prediction()
                prediction_probabilities['xgb_pred'] = pred
                prediction_probabilities['xgb_prob'] = prob
            if classifier_name in ("all", "votingclassifier"):
                voting_classifier = VotingClassifier(incident_description)
                pred, prob, _ = voting_classifier.voting_prediction()
                prediction_probabilities['voting_pred'] = pred
                prediction_probabilities['voting_prob'] = prob
            return prediction_probabilities
        # get current route
        route = request.url_rule
        if 'predictall' in route.rule:
            return predict_all()
        else:
            return make_prediction()
except ImportError:
    print("Required libraries are not installed properly, try again! ")

########### Write to CSV ###############
@app.route('/add_record', methods=['POST'])
def add_record():
    message = ""
    # Open file in append mode
    with open("/Users/melissaaranha/d2i---deakin-energy_2020t3/Datasets/ESV Data/NewIncidents.csv", 'a+', newline='') as write_obj:
        elem = []
        elem.append(str(request.form.get('action')))
        elem.append(str(request.form.get('address')))
        elem.append(str(request.form.get('Assetlbl')))
        elem.append(str(request.form.get('causecmnty')))
        elem.append(str(request.form.get('causeenv')))
        elem.append(str(request.form.get('eventDesc')))
        elem.append(str(request.form.get('causepre')))
        elem.append(str(request.form.get('techcause')))
        elem.append(str(request.form.get('causeworkp')))
        elem.append(str(request.form.get('cnttype')))
        elem.append(str(request.form.get('injtype')))
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(elem)
    return render_template('index.html', scroll="AddIncident", message="Incident added successfully!")

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, threaded=True, debug=True)