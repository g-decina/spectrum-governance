import numpy as np
import pytest
import pandas as pd
from spectrum.blue.explain import ReasonCodeGenerator # Assuming you've moved the class there

# Define the templates for the test
TEST_TEMPLATES = {
    "Credit_Score": "Credit Score of {value} is below the high-risk threshold.",
    "Time_at_Job": "Time at current job ({value} months) is too short.",
    "Debt_Ratio": "Debt-to-Income Ratio ({value}) exceeds policy limits.",
    "Age": "Age of {value} is within acceptable range (Favorable Contribution)."
}

def test_rcr_single_output_selection():
    """
    Verifies that the ReasonCodeGenerator selects the top 3 ADVERSE contributions 
    (i.e., the most negative SHAP values, representing the largest push away 
    from the favorable outcome).
    """
    rcr = ReasonCodeGenerator(templates=TEST_TEMPLATES)
    
    # Arrange: Simulate a credit denial instance (single output, e.g., Log-Odds for Approval=1)
    # The SHAP values are for the Favorable Class (Approval=1).
    # Negative values are ADVERSE contributions.
    
    feature_names = ["Credit_Score", "Time_at_Job", "Debt_Ratio", "Age", "Income"]
    
    # SHAP values for Favorable Class (Approval):
    # -2.5 (Most Adverse), -1.0 (Adverse), 0.5 (Favorable), -1.5 (Adverse), 3.0 (Most Favorable)
    shap_values = np.array([-2.5, -1.0, 0.5, -1.5, 3.0]) 
    
    feature_values = np.array([620, 10, 0.45, 45, 80000])

    # Act: Generate top 3 reasons (favorable_class_index=1 is assumed for 1D input)
    reasons = rcr.generate_reasons(
        shap_values_raw=shap_values, 
        feature_values=feature_values,
        feature_names=feature_names,
        adverse_class_index=0, # Adversary is class 0 (Denied)
        top_k=3
    )
    
    # Assert 1: Check the count
    assert len(reasons) == 3
    
    # Assert 2: Check that the selected reasons match the largest negative magnitudes.
    # Expected order: Credit_Score (-2.5), Age (-1.5), Time_at_Job (-1.0).
    assert reasons[0].startswith("Credit Score of 620.00 is")
    assert reasons[1].startswith("Feature Age contributed positively") # Fallback, Age template is favorable
    assert reasons[2].startswith("Time at current job (10 months)") 

    # We must refine the expected output based on the logic:
    # 1. phi_adverse = -shap_values (Sign flip for 1D) -> [2.5, 1.0, -0.5, 1.5, -3.0]
    # 2. adverse_indices = np.where(phi_adverse > 0) -> [0, 1, 3] (Credit, Job, Age)
    # 3. Sorted magnitudes: [2.5, 1.5, 1.0] 
    
    # Corrected Expected Order: Credit_Score (2.5), Age (1.5), Time_at_Job (1.0).
    assert reasons[0].startswith("Credit Score of 620.00 is")
    assert "Debt-to-Income" in reasons[1] # Age (1.5) uses Debt_Ratio template for Favorable
    assert "Time at current job (10 months)" in reasons[2]

    # Test the fallback
    assert "Income" not in reasons
    assert "contributed positively to the adverse score" in reasons[1] # The 'Age' feature used the fallback