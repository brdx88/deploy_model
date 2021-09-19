# Credit Risk Model
trying to deploy a Machine Learning model across from any method.

## GUIDANCE FOR INPUT AND OUTPUT FORMAT

### 1. INPUT

Your input must be JSON like this below for example:
```JSON
{
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
```

### 2. OUTPUT

The expected output should be like this below:
```PYTHON
{
    "model": "XGB-Credit-Risk",
    "prediction": "87.76% Non-default",
    "version": "1.0.0"
}
```
