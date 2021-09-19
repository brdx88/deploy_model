from flask import *
from flask.scaffold import _matching_loader_thinks_module_is_package
from predict import make_predictions
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    # return "Hello, World!"
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    # input = request.form

    # person_age = int(input['person_age'])
    # person_income = int(input['person_income'])
    # person_home_ownership = input['person_home_ownership']
    # person_emp_length = float(input['person_emp_length'])
    # loan_intent = input['loan_intent']
    # loan_grade = input['loan_grade']
    # loan_amnt = int(input['loan_amnt'])
    # loan_int_rate = float(input['loan_int_rate'])
    # loan_percent_income = float(input['loan_percent_income'])
    # cb_person_default_on_file = input['cb_person_default_on_file']
    # cb_person_cred_hist_length = int(input['cb_person_cred_hist_length'])

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
        "prediction": f"{result['data'][0]['proba']} {result['data'][0]['pred']}"

        # "score_proba":result[0],            
            # TypeError: Object of type float32 is not JSON serializable --> use str(result[0])
    }
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
