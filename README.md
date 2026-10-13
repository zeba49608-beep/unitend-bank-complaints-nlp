# unitend-bank-complaints-nlp
# Streamlining the Customer Grievance Process

An NLP project that analyses **7,011 customer complaints** received by a bank in 2023 (1 Jan to 21 Oct). It automates complaint routing with a text classifier and uses sentiment analysis to find what frustrates customers most.

> Career simulation project. Role: data scientist in the front office of a major multinational bank.

## Problem

- Complaints are triaged and reviewed manually every day, which is slow.
- Complaint data is underused for improving products and services.

## What I did

1. **Data understanding:** fixed datatypes (dates to datetime, IDs and ZIP as strings, categorical columns) and identified the date range.
2. **Preprocessing:** a `preprocessing()` function that lowercases, removes numbers, removes stopwords, removes punctuation and lemmatizes, applied to `Complaint Description`.
3. **TF-IDF:** 5,000 features (unigrams and bigrams, `min_df=5`).
4. **Classification:** predicts `Department` from the complaint text, so complaints can be routed automatically.
5. **Sentiment analysis:** VADER on the original text, plus a DistilBERT transformer on a 1,000-complaint sample.
6. **Business insights:** how the bank can use the sentiment scores.

## Results

### Department classifier (80/20 stratified split)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 74.9% | 0.72 |
| Linear SVC | 74.4% | 0.71 |

- Strongest: Mortgage (F1 0.90). Weakest: Remittance (0.61) and Others (0.57).
- Most common confusions: CASA vs Remittance, and Credit Cards vs CASA.

![Confusion matrix](images/confusion_matrix.png)

### Sentiment (VADER)

Overall: 54.5% negative, 41.0% positive, 4.5% neutral. The positive share is high because customers often write politely ("please help", "thank you") even when frustrated, so the scores are best used to **rank** complaints rather than judge any single one.

| Department | Mean compound score | Complaints ≤ -0.5 |
|---|---|---|
| Remittance | -0.321 | 54.3% |
| CASA | -0.309 | 50.6% |
| Loans | -0.132 | 40.7% |
| Credit Cards | -0.024 | 37.8% |
| Credit Reports | +0.068 | 27.2% |

![Sentiment by department](images/sentiment_by_department.png)

- **Trend:** sentiment dipped in July (-0.202) and August (-0.172) versus about -0.10 in other months.
- **Bank response:** untimely responses (-0.368) and monetary-relief cases (-0.255) are the most negative.

![Sentiment trend](images/sentiment_trend.png)

### Transformer comparison

DistilBERT results on the 1,000-complaint sample 

## How the bank can use the sentiment score

- **Priority triage:** send complaints scoring -0.5 or lower to senior agents first.
- **Churn and escalation risk:** flag very negative complaints for retention outreach.
- **Compliance monitoring:** track the share of highly negative complaints per department.
- **Product improvement:** rank products by average sentiment to find root causes.
- **Response quality:** speed matters, since untimely responses are among the most negative.
- **Early-warning dashboard:** monthly sentiment per department with drop alerts.

## Recommendations

1. Automate routing with the classifier and use sentiment as a priority flag.
2. Start root-cause reviews with Remittance and CASA.
3. Set response-time targets and alert on overdue complaints.
4. Investigate the July to August dip and build a monthly sentiment dashboard.
5. Improve the classifier on "Others" and the CASA/Remittance overlap before relying on it fully.

## Limitations

- VADER is a general-purpose tool and can miss banking language ("they charged me a fee" has no emotional words).
- Polite wording can make angry complaints look positive.
- Sentiment shows how upset a customer is, not what went wrong; pair it with classification.
- About 75% accuracy means low-confidence cases still need human review.
- October 2023 is a partial month.



## Tech stack

Python, pandas, scikit-learn, NLTK, VADER, Hugging Face Transformers, matplotlib, seaborn
