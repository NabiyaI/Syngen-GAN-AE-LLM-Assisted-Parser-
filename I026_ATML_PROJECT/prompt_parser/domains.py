from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    name: str
    numeric_columns: set[str]
    numeric_bounds: dict[str, tuple[float, float]]
    categorical_values: dict[str, set[str]]
    aliases: dict[str, str]
    phrase_to_value: dict[str, tuple[str, str]]


def _healthcare_profile() -> DomainProfile:
    return DomainProfile(
        name="healthcare_v1",
        numeric_columns={
            "age",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "glucose",
            "hba1c",
            "cholesterol_total",
            "ldl",
            "hdl",
            "triglycerides",
            "creatinine",
            "risk_score",
        },
        numeric_bounds={
            "age": (18, 90),
            "bmi": (14, 50),
            "systolic_bp": (80, 220),
            "diastolic_bp": (40, 140),
            "heart_rate": (40, 160),
            "glucose": (50, 350),
            "hba1c": (4, 15),
            "cholesterol_total": (100, 400),
            "ldl": (30, 260),
            "hdl": (20, 120),
            "triglycerides": (40, 600),
            "creatinine": (0.3, 8.0),
            "risk_score": (0, 1),
        },
        categorical_values={
            "sex": {"male", "female", "other"},
            "ethnicity": {"asian", "black", "hispanic", "white", "other"},
            "diabetes": {"yes", "no"},
            "hypertension": {"yes", "no"},
            "ckd": {"yes", "no"},
            "cad": {"yes", "no"},
            "obesity": {"yes", "no"},
            "smoker": {"yes", "no"},
            "insulin_use": {"yes", "no"},
            "antihypertensive_use": {"yes", "no"},
            "statin_use": {"yes", "no"},
            "readmission_30d": {"yes", "no"},
        },
        aliases={
            "age": "age",
            "bmi": "bmi",
            "systolic": "systolic_bp",
            "diastolic": "diastolic_bp",
            "blood pressure": "systolic_bp",
            "bp": "systolic_bp",
            "heart rate": "heart_rate",
            "glucose": "glucose",
            "blood sugar": "glucose",
            "a1c": "hba1c",
            "hba1c": "hba1c",
            "cholesterol": "cholesterol_total",
            "ldl": "ldl",
            "hdl": "hdl",
            "triglycerides": "triglycerides",
            "creatinine": "creatinine",
            "risk score": "risk_score",
            "gender": "sex",
            "sex": "sex",
            "ethnicity": "ethnicity",
            "diabetes": "diabetes",
            "hypertension": "hypertension",
            "ckd": "ckd",
            "cad": "cad",
            "obesity": "obesity",
            "smoker": "smoker",
            "insulin": "insulin_use",
            "statin": "statin_use",
            "readmission": "readmission_30d",
        },
        phrase_to_value={
            "female": ("sex", "female"),
            "females": ("sex", "female"),
            "male": ("sex", "male"),
            "males": ("sex", "male"),
            "diabetic": ("diabetes", "yes"),
            "non diabetic": ("diabetes", "no"),
            "hypertensive": ("hypertension", "yes"),
            "hypertension": ("hypertension", "yes"),
            "non smoker": ("smoker", "no"),
            "smoker": ("smoker", "yes"),
            "ckd": ("ckd", "yes"),
            "cad": ("cad", "yes"),
        },
    )


def _finance_profile() -> DomainProfile:
    return DomainProfile(
        name="finance_v1",
        numeric_columns={
            "age",
            "income",
            "credit_score",
            "debt_to_income",
            "loan_amount",
            "interest_rate",
            "tenure_months",
            "default_risk",
            "account_balance",
            "transaction_amount",
        },
        numeric_bounds={
            "age": (18, 80),
            "income": (1000, 300000),
            "credit_score": (300, 900),
            "debt_to_income": (0, 1),
            "loan_amount": (500, 1000000),
            "interest_rate": (0.01, 0.45),
            "tenure_months": (3, 360),
            "default_risk": (0, 1),
            "account_balance": (-10000, 500000),
            "transaction_amount": (1, 100000),
        },
        categorical_values={
            "employment_status": {"employed", "self_employed", "unemployed"},
            "loan_type": {"personal", "mortgage", "auto", "business"},
            "defaulted": {"yes", "no"},
            "region": {"north", "south", "east", "west"},
        },
        aliases={
            "age": "age",
            "income": "income",
            "credit score": "credit_score",
            "credit": "credit_score",
            "dti": "debt_to_income",
            "debt to income": "debt_to_income",
            "loan amount": "loan_amount",
            "interest": "interest_rate",
            "tenure": "tenure_months",
            "months": "tenure_months",
            "default risk": "default_risk",
            "balance": "account_balance",
            "transaction": "transaction_amount",
            "employment": "employment_status",
            "loan type": "loan_type",
            "defaulted": "defaulted",
            "region": "region",
            "salary": "income",
            "apr": "interest_rate",
        },
        phrase_to_value={
            "high risk": ("defaulted", "yes"),
            "defaulted": ("defaulted", "yes"),
            "non defaulted": ("defaulted", "no"),
            "mortgage": ("loan_type", "mortgage"),
            "personal loan": ("loan_type", "personal"),
            "auto loan": ("loan_type", "auto"),
            "business loan": ("loan_type", "business"),
            "employed": ("employment_status", "employed"),
            "self employed": ("employment_status", "self_employed"),
            "unemployed": ("employment_status", "unemployed"),
        },
    )


def _ecommerce_profile() -> DomainProfile:
    return DomainProfile(
        name="ecommerce_v1",
        numeric_columns={
            "age",
            "monthly_spend",
            "basket_size",
            "orders_per_month",
            "return_rate",
            "discount_usage",
            "session_duration",
            "conversion_rate",
            "customer_lifetime_value",
            "tenure_months",
        },
        numeric_bounds={
            "age": (16, 90),
            "monthly_spend": (0, 20000),
            "basket_size": (1, 50),
            "orders_per_month": (0, 60),
            "return_rate": (0, 1),
            "discount_usage": (0, 1),
            "session_duration": (1, 600),
            "conversion_rate": (0, 1),
            "customer_lifetime_value": (0, 500000),
            "tenure_months": (1, 240),
        },
        categorical_values={
            "region": {"north", "south", "east", "west"},
            "device": {"mobile", "desktop", "tablet"},
            "membership_tier": {"basic", "silver", "gold", "platinum"},
            "churned": {"yes", "no"},
        },
        aliases={
            "age": "age",
            "spend": "monthly_spend",
            "monthly spend": "monthly_spend",
            "basket": "basket_size",
            "orders": "orders_per_month",
            "returns": "return_rate",
            "discount": "discount_usage",
            "session": "session_duration",
            "conversion": "conversion_rate",
            "clv": "customer_lifetime_value",
            "ltv": "customer_lifetime_value",
            "tenure": "tenure_months",
            "membership": "membership_tier",
            "tier": "membership_tier",
            "plan": "membership_tier",
            "device": "device",
            "churn": "churned",
            "region": "region",
            "aov": "basket_size",
        },
        phrase_to_value={
            "mobile users": ("device", "mobile"),
            "desktop users": ("device", "desktop"),
            "gold members": ("membership_tier", "gold"),
            "platinum members": ("membership_tier", "platinum"),
            "churned": ("churned", "yes"),
            "non churned": ("churned", "no"),
        },
    )


DOMAIN_KEYWORDS = {
    "healthcare": "healthcare_v1",
    "medical": "healthcare_v1",
    "hospital": "healthcare_v1",
    "finance": "finance_v1",
    "banking": "finance_v1",
    "loan": "finance_v1",
    "ecommerce": "ecommerce_v1",
    "e-commerce": "ecommerce_v1",
    "retail": "ecommerce_v1",
}

DOMAIN_REGISTRY: dict[str, DomainProfile] = {
    "healthcare_v1": _healthcare_profile(),
    "finance_v1": _finance_profile(),
    "ecommerce_v1": _ecommerce_profile(),
}


def get_domain_profile(name: str) -> DomainProfile:
    if name not in DOMAIN_REGISTRY:
        raise ValueError(
            f"Unsupported domain '{name}'. Supported domains: healthcare_v1, finance_v1, ecommerce_v1"
        )
    return DOMAIN_REGISTRY[name]
