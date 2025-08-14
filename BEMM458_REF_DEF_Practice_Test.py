
#######################################################################################################################################################
# 
# Name:Riteshkumar Mahadevrao ingalepatil
# SID:740098507
# Exam Date:14/08/2025
# Module:BEMM458
# Github link for this assignment:  
#
########################################################################################################################################################
# Instruction 1. Carefully read each question before attempting the solution. Complete all tasks in the script provided.

# Instruction 2. Only ethical and minimal use of AI tools is allowed. This includes help in syntax, documentation look-up, or debugging only.
#                You must not use AI to generate the core logic or full solutions.
#                Clearly indicate where and how AI support was used.

# Instruction 3. Paste the output of each code section directly beneath it as a comment (e.g., # OUTPUT: (34, 90))

# Instruction 4. Add sufficient code comments to demonstrate your understanding of each solution.

# Instruction 5. Save your file, commit it to GitHub, and upload to ELE. GitHub commit must be done before the end of the exam session.

########################################################################################################################################################
# Question 1 - List Comprehension and String Manipulation
# You are analysing customer reviews collected from a post-service survey.
# Your SID will determine the two allocated keywords from the dictionary below. Use the **second** and **second-to-last** digits of your SID.
# For each selected keyword, identify all positions (start and end) where the word occurs in the customer_review string.
# Store each occurrence as a tuple in a list called `location_list`.

customer_review = """Thank you for giving me the opportunity to share my honest opinion. I found the packaging impressive and delivery punctual. 
However, there are several key aspects that require improvement. The installation process was somewhat confusing, and I had to refer to external 
tutorials. That said, the design aesthetics are great, and the customer support team was highly responsive. I would love to see more 
transparency in product specifications and a simpler return process. Overall, a balanced experience with clear potential for enhancement."""

# Dictionary of keywords
feedback_keywords = {
    0: 'honest',
    1: 'impressive',
    2: 'punctual',
    3: 'confusing',
    4: 'tutorials',
    5: 'responsive',
    6: 'transparent',
    7: 'return',
    8: 'enhancement',
    9: 'potential'
}

# Write your code here to populate location_list
# Example SID: 740098507  →  second digit = 4, second-to-last digit = 7
second_digit = 4  # Replace with own digits
second_last_digit = 7 # Replace with  own digits

# Get the keywords for these positions
keywords_to_find = [feedback_keywords[second_digit], feedback_keywords[second_last_digit]]

# Make list of tuples with (start, end) positions where keyword occurs
location_list = [
    (i, i + len(word))
    for word in keywords_to_find
    for i in range(len(customer_review))
    if customer_review.lower().startswith(word.lower(), i)
]

print(location_list)

# OUTPUT: [(271, 280), (458, 464)]
# Summary: In this question, I first took the second and second-to-last digits from my SID and used them as keys to pick two keywords from 
# the given dictionary. Then I wrote a list comprehension that loops through the customer review text character by character, checking if 
# each position starts with one of my keywords (case-insensitive). If it matches, I store a tuple containing the start index and the 
# position right after the keyword (end index). This means each tuple shows exactly where the keyword appears in the review. This method 
# works without having to split the sentence or manually search because it just checks all possible positions in the text directly.


########################################################################################################################################################
# Question 2 - Metrics Function for Business Intelligence
# You work in a startup focused on digital health. Your manager wants reusable functions to calculate key performance metrics:
# Gross Profit Margin, Churn Rate, Customer Lifetime Value (CLV), and Cost Per Acquisition (CPA).
# Use the **first** and **last** digits of your student ID as sample numerical values to test your function outputs.

# Insert first digit of SID here:
# Insert last digit of SID here:

# Write a function for Gross Profit Margin (%) = (Revenue - COGS) / Revenue * 100

# Write a function for Churn Rate (%) = (Customers Lost / Customers at Start) * 100

# Write a function for Customer Lifetime Value = Average Purchase Value × Purchase Frequency × Customer Lifespan

# Write a function for CPA = Marketing Cost / Number of Acquisitions

# Test your functions here
# Question 2 - Metrics Function for Business Intelligence

first_digit = 7  # First SID digit
last_digit = 7   # Last SID digit

# Gross Profit Margin (%) = (Revenue - COGS) / Revenue * 100
def gross_profit_margin(revenue, cogs):
    return (revenue - cogs) / revenue * 100

# Churn Rate (%) = (Customers Lost / Customers at Start) * 100
def churn_rate(customers_lost, customers_start):
    return (customers_lost / customers_start) * 100

# Customer Lifetime Value = Avg Purchase Value × Purchase Frequency × Customer Lifespan
def customer_lifetime_value(avg_purchase, frequency, lifespan):
    return avg_purchase * frequency * lifespan

# CPA = Marketing Cost / Number of Acquisitions
def cost_per_acquisition(marketing_cost, acquisitions):
    return marketing_cost / acquisitions

# Test with simple values based on SID digits
print("Gross Profit Margin (%):", gross_profit_margin(100, first_digit))
print("Churn Rate (%):", churn_rate(last_digit, 100))
print("Customer Lifetime Value:", customer_lifetime_value(20, first_digit, last_digit))
print("Cost Per Acquisition:", cost_per_acquisition(90, last_digit))

# OUTPUT:
# Gross Profit Margin (%): 93.0
# Churn Rate (%): 7.0
# Customer Lifetime Value: 980
# Cost Per Acquisition: 12.857142857142858
# Summary: Here I created four different functions to calculate important business metrics using the formulas given in the question. 
# I used my first and last SID digits as input values for the calculations. For example, the Gross Profit Margin function subtracts the cost 
# of goods sold from total revenue, divides by revenue, and then multiplies by 100 to get a percentage. The Churn Rate function finds the 
# percentage of customers lost compared to the starting number. The Customer Lifetime Value multiplies the average purchase value, how 
# often they buy, and how long they remain a customer. Finally, the Cost Per Acquisition divides the marketing cost by how many customers 
# were acquired. This made the code reusable for any set of inputs, not just my SID-based numbers.

########################################################################################################################################################
# Question 3 - Linear Regression for Pricing Strategy
# A bakery is studying how price affects cupcake demand. Below is a table of past pricing decisions and customer responses.
# Using linear regression:
# 1. Estimate the best price that maximises demand.
# 2. Predict demand if the bakery sets price at £25.

"""
Price (£)    Demand (Units)
---------------------------
8            200
10           180
12           160
14           140
16           125
18           110
20           90
22           75
24           65
26           50
"""

# Write your linear regression solution here
import numpy as np
from sklearn.linear_model import LinearRegression
# Price and demand data
prices = np.array([8,10,12,14,16,18,20,22,24,26]).reshape(-1, 1)
demand = np.array([200,180,160,140,125,110,90,75,65,50])

# Fit regression model
model = LinearRegression()
model.fit(prices, demand)

# Predict demand for £25
pred_demand_25 = model.predict([[25]])[0]

# Best price for max demand (linear model says lowest price is best)
best_price = prices[demand.argmax()][0]

print("Best price:", best_price)
print("Predicted demand at £25:", pred_demand_25)

# OUTPUT:
# Best price: 8
# Predicted demand at £25: 52.95454545454541
# Summary: For this question I used the scikit-learn LinearRegression model to see how the demand for cupcakes changes with price. 
# I reshaped the prices into a column format because that’s how sklearn expects the input. After fitting the model with prices and their 
# corresponding demand, I asked it to predict the demand if the price was set to £25. The model gave me around 52.95 units. I also found 
# the price that gives the highest demand by checking the position of the maximum demand in the data — in this simple case, the lowest price 
# (£8) gave the highest demand. The model shows a downward trend between price and demand, which makes sense for most products.

########################################################################################################################################################
# Question 4 - Debugging and Chart Creation
# The following code is intended to generate 100 random integers between 1 and your SID, and plot them as a scatter plot.
# However, the code contains bugs and lacks contextual annotations. Correct the code and add appropriate comments and output.

import random
import matplotlib.pyplot as plt

# Accept student ID as input
sid_input = input("740098507 ")
sid_value = int(sid_input)

# Generate 100 random values
random_values = [random.randint(1, sid_value) for _ in range(100)]

# Plotting as scatter plot
plt.figure(figsize=(10,5))
plt.scatter(range(100), random_values, color='green', marker='x', label='Random Values')
plt.title("Scatter Plot of 100 Random Numbers")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

Best price: 8
Predicted demand at £25: 52.95454545454541 output: Scatter plot appears with 100 points in green 'x'.