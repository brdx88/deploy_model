from flask import *
from flask.scaffold import _matching_loader_thinks_module_is_package
from predict import make_predictions
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    # return "Hello, World!"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
