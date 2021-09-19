from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # library for handling cross origin resources sharing.
from predict import make_predictions


def create_app():
    """ app factories """
    app = Flask(__name__)
    CORS(app)


    @app.route("/", methods=["GET"])
    def default():
        return render_template("index.html")


    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            data_input = request.get_json()["data"]
            data = {}
           
            data['person_age'] = int(data_input.get('person_age'))
            data['person_income'] = int(data_input.get('person_income'))
            data['person_home_ownership'] = data_input.get('person_home_ownership')
            data['person_emp_length'] = float(data_input.get('person_emp_length'))
            data['loan_intent'] = data_input.get('loan_intent')
            data['loan_grade'] = data_input.get('loan_grade')
            data['loan_amnt'] = int(data_input.get('loan_amnt'))
            data['loan_int_rate'] = float(data_input.get('loan_int_rate'))
            data['loan_percent_income'] = float(data_input.get('loan_percent_income'))
            data['cb_person_default_on_file'] = data_input.get('cb_person_default_on_file')
            data['cb_person_cred_hist_length'] = int(data_input.get('cb_person_cred_hist_length'))


            result = make_predictions(data)
            
            result = {
                        'model':'XGB-Credit-Risk',
                        "version": '1.0.0',
                        "prediction": f"{result['data'][0]['proba']} {result['data'][0]['pred']}"

                        # "score_proba":result[0],            
                            # TypeError: Object of type float32 is not JSON serializable --> use str(result[0])
                    }
            
        return jsonify(result)

    @app.route('/predict-api', methods=['POST'])
    def predict_api():
        data = request.get_json()
        result = make_predictions(data)

        result = {
			'model':'XGB-Credit-Risk',
            "version": '1.0.0',
            "prediction": f"{result['data'][0]['proba']} {result['data'][0]['pred']}"
	  	}	
        print(result)
        return jsonify(result)
    return app

