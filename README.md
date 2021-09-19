# Credit Risk Model

## Business Problems
As a Financing Company, the user wants to build a credit scoring model to predict whether the client will default or not after their loan application.

## Business Goals
Research and develop the model to predict applicants whether the applicant will default or not, and also find the best metrics since this is an imbalance class dataset.

## Results
```bash
    +----------------------+--------------------------------+--------------------------------+--------------------------------+
    |        		   | Train	    	   	    | Test	    	   	     | Holdout Sample	   	      | 
    | Model                +----------+----------+----------+----------+----------+----------+----------+----------+----------+
    |   		   | Recall   | F1-Score | AUC	    | Recall   | F1-Score | AUC	     | Recall   | F1-Score | AUC      |
    +----------------------+----------+----------+----------+----------+----------+----------+----------+----------+----------+	
    | Logistic Regression  | 0.525296 | 0.618740 | 0.737900 | 0.524548 | 0.621112 | 0.738705 | 0.470407 | 0.584248 | 0.717754 |
    | RandomForest	   | 0.000000 | 0.000000 | 0.500000 | 0.000000 | 0.000000 | 0.500000 | 0.000000 | 0.000000 | 0.500000 |
    | XGBoost  		   | 0.695587 | 0.813239 | 0.845633 | 0.689061 | 0.798403 | 0.839225 | 0.696387 | 0.802480 | 0.843304 |
    +----------------------+----------+----------+----------+----------+----------+----------+----------+----------+----------+
```

## Conclusions
Since this case is an imbalanced dataset `(non-default:77.7% ; default:22.3%)`, it's worth looking at the AUC and Recall metrics instead. Why? Especially for Recall metrics. For business purposes, we assume to minimize Type 2 (minimize False Negative -- *predict non-default (0), actual default (1)*). Hence, we use Recall metrics for optimum result.

It can be seen in the table above, the model which has the highest and the most stable AUC and Recall is `XGBoost AUC: 0.839225` and `XGBoost Recall: 0.689061`.

In addition, the Recall and AUC scores on the train and test are not much different. It means that we can conclude that this model is 'just right' to classify `target 1` and `target 0`, neither overfitting nor underfitting.

If we look back at the features importance by Logistic Regression with Lasso regularization, the selected features seem make sense. Features which affect `loan_status` are:
1. Percentage of Income *`('loan_percent_income')`*, 
1. Loan Amount *`('loan_amnt_WOE')`*,
9. Employement Length *`('person_emp_length_WOE')`*,
3. Owning Home *`('person_home_ownership_OWN')`*,
5. Loan Grade *`('loan_grade')`*,
4. Intention for Venture *`('loan_intent_VENTURE')`*, 
6. Intention for Education *`('loan_intent_EDUCATION')`*, 
7. Renting home *`('person_home_ownership_RENT')`*,
2. Age *`('person_age_WOE')`*,
1. Credit History Length *`('cb_person_cred_hist_length_WOE')`*,
8. Intention for personal purposes *`('loan_intent_PERSONAL')`*,         
10. Intention for home improvement *`('loan_intent_HOMEIMPROVEMENT')`*.

After tuning the models and get each metrics, we could predict the holdout sample using our previous models. We see that the XGBoost algorithm shows its best performance among the others. In the holdout sample, XGBoost can reach the `AUC: 0.843304` and the `Recall: 0.696387`. It tells us that XGBoost could be our model for production, because it's not overfitted and it can predicts the holdout sample very well.

# GUIDANCE FOR INPUT AND OUTPUT FORMAT
Guidance for input and output format when access it on web.

### 1. Input Format
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

### 2. Output Format
The expected output should be like this below:
```PYTHON
{
    "model": "XGB-Credit-Risk",
    "prediction": "87.76% Non-default",
    "version": "1.0.0"
}
```
