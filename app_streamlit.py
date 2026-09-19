import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(payload):
    runtime = get_runtime_client()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    return json.loads(response["Body"].read().decode("utf-8"))

st.set_page_config(page_title="Credit Score Classification", layout="wide")
st.title("Credit Score Classification System")
st.write("Provide the customer's financial and credit-related information to generate a credit score prediction.")

with st.form("prediction_form"):
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    occupation = st.selectbox("Occupation",
        [
            "Accountant",
            "Architect",
            "Developer",
            "Doctor",
            "Engineer",
            "Entrepreneur",
            "Journalist",
            "Lawyer",
            "Manager",
            "Mechanic",
            "Media_Manager",
            "Musician",
            "Scientist",
            "Teacher",
            "Writer"
        ]
    )

    st.subheader("Income & Financial Profile")
    annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)
    invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=300.0)
    monthly_balance = st.number_input("Monthly Balance", min_value=-1000000.0, max_value=1000000.0,value=500.0)
    
    st.subheader("Banking & Credit Accounts")
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=50, value=4)
    num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=50, value=3)
    credit_utilization = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=100.0, value=30.0, help="Percentage of available credit currently being used.")
    credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0.0, max_value=100.0, value=4.0)
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, max_value=50.0, value=10.0, help="Percentage change in the customer's credit limit.")
    
    st.subheader("Loans & Debt Information")
    num_loans = st.number_input("Number of Loans", min_value=0, max_value=50, value=2)
    selected_loans = st.multiselect("Select Loan Types",
        [
            "Auto Loan",
            "Credit-Builder Loan",
            "Debt Consolidation Loan",
            "Home Equity Loan",
            "Mortgage Loan",
            "Not Specified",
            "Payday Loan",
            "Personal Loan",
            "Student Loan",
            "Unknown"
        ],
        help="Select all loan types currently held by the customer. If loan information is unavailable, select 'Unknown' only."
    )
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
    total_emi = st.number_input("Total EMI per Month", min_value=0.0, value=200.0)
    interest_rate = st.number_input("Interest Rate", min_value=0, max_value=100, value=10)

    st.subheader("Payment Behaviour")
    delay_due = st.number_input("Delay From Due Date", min_value=0, max_value=100, value=10)
    delayed_payment = st.number_input("Number of Delayed Payments", min_value=0.0, max_value=100.0, value=5.0)
    payment_min_amount = st.selectbox("Payment of Minimum Amount", ["No", "Yes"], help="Indicates whether the customer usually pays only the minimum required amount.")
    payment_behaviour = st.selectbox("Payment Behaviour",
        [
            "High_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "Low_spent_Small_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments"
        ],
        help="Represents the customer's spending and payment behavior pattern."
    )

    st.subheader("Credit History & Credit Quality")
    credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"], help="Credit Mix represents the variety of credit accounts owned by the customer.")
    credit_history_years = st.number_input("Credit History Years", min_value=0, max_value=50, value=5, help="Length of time the customer has maintained credit accounts.")
    credit_history_months = st.number_input("Additional Months", min_value=0, max_value=11, value=0)
    
    st.subheader("Additional Information")    
    month = st.selectbox("Month",
        [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ],
        help="Select the month when the customer's financial data was recorded."
    )

    submitted = st.form_submit_button("Predict Credit Score")

if submitted:
    if len(selected_loans) == 0:
        st.error("Please select at least one loan type.")
        st.stop()

    if ("Unknown" in selected_loans and len(selected_loans) > 1):
        st.error("Unknown cannot be selected together with other loan types.")
        st.stop()

    credit_history_age = (credit_history_years * 12 + credit_history_months)

    data = {
        "Month": month,
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_due,
        "Num_of_Delayed_Payment": delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_min_amount,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested_monthly,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
        "Auto Loan": 0,
        "Credit-Builder Loan": 0,
        "Debt Consolidation Loan": 0,
        "Home Equity Loan": 0,
        "Mortgage Loan": 0,
        "Not Specified": 0,
        "Payday Loan": 0,
        "Personal Loan": 0,
        "Student Loan": 0,
        "Unknown": 0
    }

    for loan in selected_loans:
        data[loan] = 1

    try:
        result = invoke_endpoint(data)
        prediction = result["label"]
        probs = result["probabilities"]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "Good":
            st.success("### Credit Score: Good")
            st.info("Customer has a strong credit profile.")
        elif prediction == "Standard":
            st.warning("### Credit Score: Standard")
            st.info("Customer has an average credit profile.")
        else:
            st.error("### Credit Score: Poor")
            st.info("Customer may have a higher credit risk.")

        st.divider()
        st.subheader("Prediction Probabilities")

        for class_name, prob in probs.items():
            st.write(f"**{class_name}** : {prob:.1%}")
            st.progress(float(prob))

    except NoCredentialsError:
        st.error("AWS credentials not found.")

    except ClientError as e:
        st.error(e.response["Error"]["Message"])