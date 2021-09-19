import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

raw_input = {
	"person_age":24,
	"person_income":168000,
	"person_home_ownership":"MORTGAGE",
	"person_emp_length":0.0,
	"loan_intent":"PERSONAL",
	"loan_grade":"E",
	"loan_amnt":25000,
	"loan_int_rate":16.45,
	"loan_percent_income":0.15,
	"cb_person_default_on_file":"N",
	"cb_person_cred_hist_length":3
	}

with open("xgb_piped_preps_only", "rb") as f:
	fe_pipe = pickle.load(f)
with open("xgb_piped_model_only", "rb") as f:
	model_pipe = pickle.load(f)

def formatting_data(raw_input):
	raw_input = pd.DataFrame.from_dict(raw_input, orient='index').T.replace({
		None : np.nan,
		"null": np.nan,
		"": np.nan
	})
	return raw_input

def preprocess(data):
	result = fe_pipe.transform(data)
	return result

def make_predictions(data):
	data = formatting_data(data)
	data = preprocess(data)
	
	pred = model_pipe.predict(data)
	proba = model_pipe.predict_proba(data)

	if pred == 0:
		pred = "Non-default"
		proba = f"{round(proba[0][0]*100, 2)}%"
		print(proba, pred)
		result = {"data" : [ {'proba' : proba, 'pred' : pred}]}
		return result
	else:
		pred =  "Default"
		proba = f"{round(proba[0][1]*100, 2)}%"
		print(proba, pred)
		result = {"data" : [ {'proba' : proba, 'pred' : pred}]}
		return result

if __name__ == "__main__":
	result = make_predictions(raw_input)
	print(type(result))
	print(result)

