[README.md](https://github.com/user-attachments/files/30637634/README.md)
# Capstone-Project---Phase-1-EDA-
UC Berkeley Engineering - Machine Learning and Artificial Intelligence - Capstone Project 

# Using Machine Learning to Identify Socioeconomic and Behavioral Drivers of Chronic Disease Risk in U.S. Adults

**Machine Learning & Artificial Intelligence — Capstone Project**
Vithy Vithyanandan · Updated - August 2, 2026

## Project Overview

For my capstone project, I plan to use CDC BRFSS data to predict chronic disease outcomes among U.S. adults based on socioeconomic, demographic, access-to-care, and behavioral risk factors. The project explores whether variables such as income, education, insurance status, physical activity, smoking, BMI, and diet can help identify individuals at higher risk for conditions such as diabetes, hypertension, cardiovascular disease, or obesity.

## Data Needed

The primary data source is the CDC Behavioral Risk Factor Surveillance System, which includes self-reported survey data from adults across the United States. Chronic disease indicators serve as outcome variables, with predictors including age, sex, race/ethnicity, education, income, employment, insurance coverage, smoking status, alcohol use, physical activity, BMI, diet, sleep, general health status, and access-to-care measures. Depending on final scope, the target outcome is diabetes, hypertension, obesity, or a combined chronic disease indicator.

> **Dataset:** CDC BRFSS (Behavioral Risk Factor Surveillance System) — a massive annual survey (400,000+ respondents) covering health behaviors and demographics.

- The BRFSS is the nation's premier system of health-related telephone surveys collecting uniform, state-specific data about U.S. residents' health-related risk behaviors, chronic health conditions, and use of preventive services.
- The objective of the BRFSS is to gather consistent, state-level data on preventive health practices and risk behaviors associated with chronic diseases, injuries, and preventable infectious diseases among adults (aged 18 and older).

## Potential Modeling Approaches

1. **Logistic regression** — provides a baseline predictive model and helps estimate the relationship between socioeconomic/behavioral variables and the likelihood of a chronic disease outcome.
2. **Decision tree or random forest classification** — captures nonlinear relationships and interactions between predictors, such as how age, income, BMI, smoking, and physical activity together affect chronic disease risk.

## Expected Results

The models are expected to identify a meaningful set of socioeconomic and behavioral predictors — such as income, BMI, smoking status, physical activity, and insurance coverage — that are significantly associated with chronic disease risk among U.S. adults. Logistic regression is expected to serve as a strong, interpretable baseline, producing odds ratios that quantify each predictor's contribution to chronic disease probability. Decision tree and random forest models are expected to outperform logistic regression in overall predictive accuracy by capturing nonlinear relationships and variable interactions (e.g., the combined effect of low income, physical inactivity, and high BMI on diabetes risk) that linear models may miss.

## Why This Question Is Important

Chronic diseases such as diabetes, hypertension, cardiovascular disease, and obesity represent the leading causes of mortality and healthcare expenditure in the United States. Despite their prevalence, chronic disease risk is not evenly distributed — it is deeply shaped by socioeconomic conditions, access to care, and behavioral patterns that are often addressable through targeted intervention.

Traditional approaches to identifying at-risk populations rely heavily on clinical screenings, which require individuals to already be engaged with the healthcare system. This creates a critical gap: individuals who are uninsured, lower-income, or who face structural barriers to healthcare access are precisely those least likely to receive timely screening — and most likely to bear elevated chronic disease risk.

Machine learning offers an opportunity to close this gap by identifying high-risk individuals using population-level survey data, before clinical symptoms appear. By applying predictive models to BRFSS data, this project can help surface the specific combination of socioeconomic and behavioral factors that drive risk at scale, informing where public health resources and prevention programs should be directed. This is especially valuable for policymakers, health departments, and community organizations working to reduce health disparities across income, race, and geography in the U.S. adult population.

## Exploratory Data Analysis

### Data Source and Preparation

The analysis uses the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 public release (`LLCP2024`), the most recent annual cycle available. From the full survey, 21 variables were selected covering the diabetes target along with weight/BMI, cardiometabolic history, health-access, behavioral, and demographic predictors. The initial extract contained **456,636 respondents**.

The original four-category target (`DIABETE4`) was collapsed into a binary outcome, `DIABETE_BINARY`, by combining diagnosed diabetes and borderline/prediabetes into the positive class (1) and "no diabetes" into the negative class (0); "don't know" and "refused" responses were treated as missing.

### Missing Data

Missingness varied considerably across variables:

| Variable | Missing % |
|---|---|
| `INCOME3` (income) | 19.0% |
| `_RFDRHV9` (heavy drinker) | 10.2% |
| `_BMI5` / `_RFBMI5` (BMI) | 9.4% |
| `SMOKE100` (smoking) | 6.9% |
| `DIFFWALK` (difficulty walking) | 4.4% |

Most other predictors were missing in under 5% of records, and the target variable had no missing values. At the row level, **65.7%** of respondents had complete data across all 21 variables, while only **9.7%** were missing three or more fields — indicating missingness was concentrated in a relatively small subset of respondents rather than spread evenly.

Rows with any missing value were dropped for this initial modeling pass, reducing the dataset from 456,636 to **300,031 respondents** (a 34.3% reduction), driven mainly by the small number of high-missingness columns above rather than by missingness scattered randomly across the dataset.

### Class Balance

After removing incomplete records, the cleaned dataset shows a target split of **83.4% non-diabetic** (250,118 respondents) versus **16.6% diabetic/prediabetic** (49,913 respondents) — an imbalance ratio of roughly **5:1**. This confirms the need for class-imbalance-aware modeling strategies in Phase 2 (e.g., stratified train/test splits, class weighting, or resampling) and supports prioritizing F1, ROC-AUC, precision, and recall over raw accuracy when evaluating models.

### Continuous Variables

- **BMI (`_BMI5`)** averages 28.8 (SD 6.6), placing the typical respondent in the overweight range, with values ranging from 12 to just under 100.
- **Physical and mental health** (`PHYSHLTH`, `MENTHLTH`, each "bad days out of the past 30") are both heavily right-skewed: the median respondent reports zero bad days in both categories, but the mean of ~4.4 days per measure indicates a smaller subset of respondents driving most of the burden.
- A BMI density plot split by diabetes status shows the diabetic/prediabetic group shifted noticeably toward higher BMI values relative to the non-diabetic group.

### Categorical Variables and Risk Factor Patterns

- Diabetes/prediabetes prevalence **rises steadily with age**, climbing from the lowest rates in the 18–24 group to the highest in the 70+ groups.
- Prevalence **increases monotonically as self-reported general health worsens**, moving from "Excellent" through "Poor."
- Across nine binary risk indicators (stroke history, heart disease/MI, kidney disease, smoking, heavy drinking, physical activity, difficulty walking, insurance coverage, and cost-related care avoidance), respondents reporting **stroke history, heart disease, kidney disease, and difficulty walking** show visibly higher diabetes/prediabetes rates, while **physical activity** is associated with lower rates.
- **Income and education** both show an inverse relationship with diabetes/prediabetes prevalence, with rates generally declining as income bracket and education level increase.
- Rate comparisons by **sex** show a modest difference between males and females, and rates by **race/ethnicity** vary across groups, warranting further fairness-aware evaluation once modeling begins.

### Correlation Analysis

A full correlation matrix was computed across all numeric predictors and the target.

**Most positively correlated with the target:**
- General health status (r = 0.267)
- Age group (r = 0.221)
- BMI (r = 0.198)
- Physical-health bad days (r = 0.156)
- Physical activity (r = 0.149)

**Most negatively correlated with the target:**
- Difficulty walking (r = -0.216)
- Heart disease/MI (r = -0.177)
- Kidney disease (r = -0.168)
- Income (r = -0.143)
- Routine checkup recency (r = -0.128)

> **Note:** several binary risk-factor columns remain in their raw BRFSS coding (1 = Yes, 2 = No) at this EDA stage, which inverts the intuitive sign of their correlation with the target. These will be re-encoded during the Phase 2 modeling pipeline for cleaner interpretation.

### Summary of Phase 1 EDA

Overall, the exploratory analysis confirms that **age, general health status, BMI, mobility limitations, and chronic condition history** are the strongest linear signals associated with diabetes/prediabetes risk in this cohort, consistent with established clinical and public health literature. The pronounced class imbalance (~83/17) and moderate missingness concentrated in a handful of variables are the two key data-quality considerations carried into Phase 2 modeling.

## Next Steps (Phase 2)

- Address class imbalance via stratified splits, class weighting, and/or resampling
- Re-encode raw BRFSS binary variables to standard 0/1 for cleaner model coefficients and correlation signs
- Build and compare classification models (logistic regression, decision tree/random forest, ensemble methods)
- Evaluate using F1, ROC-AUC, precision, and recall rather than raw accuracy
