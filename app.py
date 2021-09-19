from flask import *
from flask.scaffold import _matching_loader_thinks_module_is_package
from predict import make_predictions
from xgboost import XGBClassifier
# from flask_cors import CORS

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    # return "Hello, World!"
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
	    data_input = request.get_json()
	    data = {}

	    data["person_age"] = int(data_input['data']['person_age'])
	    data["person_income"] = int(data_input['data']['person_income'])
	    data["person_home_ownership"] = data_input['data']['person_home_ownership']
	    data["person_emp_length"] = float(data_input['data']['person_emp_length'])
	    data["loan_intent"] = data_input['data']['loan_intent']
	    data["loan_grade"] = data_input['data']['loan_grade']
	    data["loan_amnt"] = int(data_input['data']['loan_amnt'])
	    data["loan_int_rate"] = float(data_input['data']['loan_int_rate'])
	    data["loan_percent_income"] = float(data_input['data']['loan_percent_income'])
	    data["cb_person_default_on_file"] = data_input['data']['cb_person_default_on_file']
	    data["cb_person_cred_hist_length"] = int(data_input['data']['cb_person_cred_hist_length'])

	    result = make_predictions(data)
	    result = {
                    'model':'XGB-Credit-Risk',
                    "version": '1.0.0',
                    "prediction": f"{result['data'][0]['proba']} {result['data'][0]['pred']}"
		  	}

	    return jsonify(result)


    # data = {
    #     "person_age":person_age,
    #     "person_income":person_income,
    #     "person_home_ownership":person_home_ownership,
    #     "person_emp_length":person_emp_length,
    #     "loan_intent":loan_intent,
    #     "loan_grade":loan_grade,
    #     "loan_amnt":loan_amnt,
    #     "loan_int_rate":loan_int_rate,
    #     "loan_percent_income":loan_percent_income,
    #     "cb_person_default_on_file":cb_person_default_on_file,
    #     "cb_person_cred_hist_length":cb_person_cred_hist_length
	# }

    data = request.get_json()
    result = make_predictions(data)
    result = {
        'model':'XGB-Credit-Risk',
        "version": '1.0.0',
		'score_proba': result['data'][0]['proba'],
		'prediction': result['data'][0]['pred'],
		'result': str(round(result['data'][0]['proba'], 3))

        # "score_proba":result[0],            
            # TypeError: Object of type float32 is not JSON serializable --> use str(result[0])
    }
    print(result)
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



if __name__ == '__main__':
    app.run(port=5000, debug=True)
