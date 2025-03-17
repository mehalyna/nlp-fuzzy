# Customer Churn Prediction

## Existing articles on Fuzzy Utility

- [A novel cost-sensitive framework for customer churn predictive modeling](https://www.econstor.eu/bitstream/10419/161801/1/10.1186-s40165-015-0014-6.pdf#:~:text=the%20models%20miss%20to%20include,detect%20a%20real%20churner%20versus)
- 
- 
- 
- 
- 
- 
- 
- 


## Plan

**1. EDA**

*Objective:*

Understand customer behaviors influencing churn and define variables critical for the utility function.

*Actions:*

- Descriptive Statistics: Mean, median, distribution analysis (e.g., age, income, spending).
- Correlation analysis: Identify variables strongly correlated with churn.
- Segment customers based on behaviors (purchasing patterns, returns, promotions response, etc.).

*Utility-Focused Analysis:*

- Determine each customer’s approximate lifetime value (CLV) from columns like: Annual_Income, Total_Spend, Years_as_Customer, and Average_Transaction_Amount.
- Identify features most correlated with churn (Satisfaction_Score, Num_of_Returns, Num_of_Support_Contacts, Last_Purchase_Days_Ago) to quantify risk and utility.



**2. Define Customer Utility Functions**

*Objective*: Create a mathematical definition of customer utility based on <span style="color:red">Cumulative Prospect Theory (CPT)</span>.

*Customer Utility Function (Example)*: 

$$U(Customer) = w^+(P_{\text{stay}})\times V(\text{Gain}) - w^-(P_{\text{churn}})\times V(\text{Loss})$$

- Reference point (r): Current satisfaction or status quo (use Satisfaction_Score).
- Gain: The perceived benefit from staying (continued good experience, incentives, convenience).
- Loss: The perceived benefits lost by leaving (loyalty points, personalized offers).

Practical Implementation:

- Estimate the parameters (e.g., loss aversion coefficient λ, probability weighting functions w(•)) from historical data.
- Translate the impact of Satisfaction_Score, Promotion_Response, and Num_of_Support_Contacts into quantifiable utilities.

**3: Data Preprocessing & Feature Engineering**

*Encode categorical variables:* Gender, Promotion_Response, and Email_Opt_In using dummy variables or one-hot encoding.

*Create engineered features like:*

- Ratio of returns (Num_of_Returns / Num_of_Purchases)
- Frequency of purchase (Num_of_Purchases/Last_Purchase_Days_Ago)
- Engagement score combining Promotion_Response and Email_Opt_In

*Calculate Customer Utility Metrics:*

- Compute expected CLV for each customer.
- Compute estimated cost per customer intervention (e.g., marketing cost per offer).

**4: Machine Learning with Utility Functions**

***Approach A: Cost-sensitive Classification (Decision Trees / Random Forest)***

Simple and interpretable initial approach to immediately leverage utility.

Process:

- Define utility (cost-sensitive) matrix:

| Prediction / Reality | Churn (True) | Stay (False) |
  |----------------------|--------------|--------------|
  | Predict Churn        | +High utility (CLV-cost)   | -Moderate cost |
  | Predict Stay         | -High loss (lost CLV)      | +Neutral utility |

- Train decision tree model (e.g., scikit-learn DecisionTreeClassifier with class weights based on utility).

*Expected outcome:* Quickly identify high-utility intervention customers, reducing unnecessary promotional costs and maximizing CLV retained.



***Approach B: Bayesian Decision Model (Bayes Minimum Risk)***


Explicitly integrates probability of churn with utility, optimizing intervention decisions based on expected outcomes.

*Procedure:*

- Compute posterior churn probabilities (P(churn)) using a Bayesian Classifier (e.g., Naive Bayes or Bayesian Network).

- Decision Rule:

Intervene if:
$E[Utility] = P(churn)\times CLV - Cost_{intervention} > 0$

*Expected Benefit:* Optimizes spend by selecting customers for whom intervention provides positive expected value.

<span style="color:red">Bayes Minimum Risk</span>

***Approach C: Reinforcement Learning (Contextual Bandits)***

Continuously learns optimal actions dynamically—particularly effective in a setting with frequent customer interactions.

RL Framework:

- State: Customer state represented by dataset features (e.g., recency, frequency, monetary value, satisfaction).
- Action: Promotional incentives (discounts, loyalty benefits, emails).
- Reward (Utility):
Positive reward if customer retained (CLV – Cost).
Negative if churned after intervention (- Cost).

*Contextual Bandit implementation:*

- Start simple by testing different promotions.
- Continuously update action-utility pair based on observed responses, converging to the best action for each segment.

<span style="color:red">Contextual Bandits</span>


**5: Utility-Based Promotional Targeting**


- Prioritize customers based on highest expected incremental retention utility (`Churn Risk × CLV – Cost`).
- Focus resources on customers with high churn risk and high expected response to intervention (e.g., customers who previously responded to promotions).

*Practical Recommendation Example:*

- High CLV, moderate churn probability, high response likelihood → Target aggressively (valuable segment, cost-effective).
- Low CLV, high churn probability, low response likelihood → Minimal/no targeting (wasteful resources).


*Expected Outcomes & Evaluation Metrics*

- Measure effectiveness through clear KPIs:
`Primary`: % churn reduction.
`Secondary`: Increase in retention ROI, targeted campaign effectiveness.

*Realistic targets:*

- Cost-sensitive/Bayesian: 15–25% churn reduction.
- Reinforcement Learning (if effectively deployed): 25–40% churn reduction.

*Next steps:*

- Begin immediately with cost-sensitive models to identify high-value, at-risk customers.
- Pilot Bayesian Minimum Risk models to optimize campaign targeting effectiveness further.
- Explore Contextual Bandits for continuous optimization once infrastructure and skills are ready.
- Regularly review model outcomes and adjust utility parameters for maximum effectiveness.


**6. More on ML with Utility Theory!**

*Applying Specific Utility Theory Variant: Prospect Theory (CPT)*

$$U(x) = \begin{cases} (x - r)^{\alpha}, & x \geq r \\ -\lambda (r - x)^{\beta}, & x < r \end{cases}$$

- $x$ = outcome (e.g., perceived monetary or psychological benefit)
- $r$ = reference point (current customer experience)
- $\alpha$, $\beta$ represent sensitivity parameters for gains and losses
- $\lambda$ represents loss-aversion (typically >1.0)

