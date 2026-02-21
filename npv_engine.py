import numpy as np
import pandas as pd
import time
import pickle
import os

class AccountNPVEngine:
    """
    Highly efficient Vectorized Account-Level NPV Model Engine.
    Uses numpy broadcasting to process millions of accounts without Python loops.
    Attributes are configured as floats32 to optimize memory footprint and CPU cache usage.
    """
    def __init__(self, num_months=99, annual_discount_rate=0.08, rdm_model_path=None, curve_model_paths=None):
        self.num_months = num_months
        
        # Base annual discount rate, will be combined with annual loss rate dynamically
        self.base_annual_discount_rate = np.float32(annual_discount_rate)

        # Load external models if provided (LightGBM, sklearn, etc.)
        self.rdm_model = self._load_model(rdm_model_path) if rdm_model_path else None
        
        if curve_model_paths and len(curve_model_paths) == 4:
            self.curve_models = [self._load_model(p) for p in curve_model_paths]
        else:
            self.curve_models = None
        
        # 1. Initial Regression Model (RDM) Configurations
        # We assume 5 features. These are dummy regression weights for the Risk Driver Metric.
        self.rdm_coefs = np.array([0.45, -0.15, 0.22, 0.05, -0.30], dtype=np.float32)
        self.rdm_intercept = np.float32(1.2)
        
    def _load_model(self, path):
        """Helper to load a pickle file or return the object if it's already a model."""
        if hasattr(path, 'predict'):
            return path
        if isinstance(path, str) and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
        
    def score_initial_rdm_model(self, rdm_features):
        """
        Scores the initial RDM regression model using 5 characteristics.
        Produces a single number output per account.
        
        :param rdm_features: shape (num_accounts, 5)
        :return rdm: shape (num_accounts,)
        """
        if self.rdm_model is not None:
            # Predict using the loaded scikit-learn or LightGBM model
            return self.rdm_model.predict(rdm_features).astype(np.float32)
            
        # Vectorized dot product across all accounts (Fallback)
        return np.dot(rdm_features, self.rdm_coefs) + self.rdm_intercept

    def score_curve_models(self, rdm, other_features):
        """
        Scores 4 distinct regression models that produce curves over 99 months.
        Using RDM, other characteristics, and month on book (1 to 99).
        
        :param rdm: shape (num_accounts,)
        :param other_features: shape (num_accounts, num_other_features)
        :return c1, c2, c3, c4: curves each of shape (num_accounts, 99)
        """
        num_accounts = rdm.shape[0]
        
        # Combine RDM output and other characteristics into a single feature matrix
        base_features = np.column_stack((rdm, other_features))
        num_base_features = base_features.shape[1]
        
        # Repeat each account's features 99 times
        # Shape: (num_accounts * 99, num_base_features)
        repeated_features = np.repeat(base_features, self.num_months, axis=0)
        
        # Create an array for month on book (1 to 99) repeated for each account
        # Shape: (num_accounts * 99,)
        mob_array = np.tile(np.arange(1, self.num_months + 1, dtype=np.float32), num_accounts)
        
        # Combine the repeated features with the month on book
        # Shape: (num_accounts * 99, num_base_features + 1)
        expanded_features = np.column_stack((repeated_features, mob_array))
        
        # Convert to a DataFrame as the models (like GBM) often expect feature names or a DataFrame structure
        feature_cols = [f"feature_{i}" for i in range(num_base_features)] + ["month_on_book"]
        df_features = pd.DataFrame(expanded_features, columns=feature_cols)
        
        # Calculate actual curve values for each account-month observation
        if self.curve_models is not None:
            # Use loaded models (LightGBM / sklearn) predicting directly over the dataframe
            flat_curve_1 = self.curve_models[0].predict(df_features).astype(np.float32)
            flat_curve_2 = self.curve_models[1].predict(df_features).astype(np.float32)
            flat_curve_3 = self.curve_models[2].predict(df_features).astype(np.float32)
            flat_curve_4 = self.curve_models[3].predict(df_features).astype(np.float32)
        else:
            # Fallback to dummy matrix dot product (simulating the models predicting the actual curves)
            np.random.seed(42)  # Fixed for dummy consistency
            coef_1 = np.random.randn(num_base_features + 1).astype(np.float32) * 0.01
            coef_2 = np.random.randn(num_base_features + 1).astype(np.float32) * 0.05
            coef_3 = np.random.randn(num_base_features + 1).astype(np.float32) * 0.02
            coef_4 = np.random.randn(num_base_features + 1).astype(np.float32) * 1.5
            
            flat_curve_1 = np.dot(expanded_features, coef_1)
            flat_curve_2 = np.dot(expanded_features, coef_2) 
            flat_curve_3 = np.dot(expanded_features, coef_3)
            flat_curve_4 = np.dot(expanded_features, coef_4)
        
        # Reshape the flat predictions back into (num_accounts, 99) matrices
        curve_1 = flat_curve_1.reshape(num_accounts, self.num_months)
        curve_2 = flat_curve_2.reshape(num_accounts, self.num_months)
        curve_3 = flat_curve_3.reshape(num_accounts, self.num_months)
        curve_4 = flat_curve_4.reshape(num_accounts, self.num_months)
        
        return curve_1, curve_2, curve_3, curve_4

    def calculate_cashflows(self, c1, c2, c3, c4):
        """
        Combines curves using mathematical formula to calculate 99 month cashflow
        
        :param c1, c2, c3, c4: curves of shape (num_accounts, 99)
        :return cashflows: shape (num_accounts, 99)
        """
        # A simple illustration of mathematical formula using the 4 model outputs
        # Evaluates element-wise across the matrices (num_accounts x 99)
        # E.g., Cashflow = Balance(c2) * Rate - PD(c1)*Balance(c2) + Fees(c4) - Prepay(c3)*Balance(c2)
        cashflows = (c2 * 0.02) - (c1 * c2) + c4 - (c3 * c2)
        return cashflows

    def calculate_annual_loss_rate(self, c1, c2):
        """
        Calculate a 1-year (first 12 months) annualized loss rate for each account.
        Assumes c1 is PD (loss rate) and c2 is Balance.
        """
        months_1yr = min(12, self.num_months)
        c1_1yr = c1[:, :months_1yr]
        c2_1yr = c2[:, :months_1yr]
        
        avg_bal_1yr = np.mean(c2_1yr, axis=1)
        safe_avg_bal = np.where(avg_bal_1yr == 0, 1e-9, avg_bal_1yr)
        
        losses_1yr = np.sum(c1_1yr * c2_1yr, axis=1)
        annual_loss_rate = losses_1yr / safe_avg_bal
        return annual_loss_rate

    def calculate_npv(self, cashflows, annual_loss_rate):
        """
        Discount 99-month cashflows to get a single NPV number per account.
        Discount rate is dynamically calculated per account: base_rate + annual_loss_rate
        
        :param cashflows: shape (num_accounts, 99)
        :param annual_loss_rate: shape (num_accounts,)
        :return npv: shape (num_accounts,)
        """
        # Calculate dynamic discount rate for each account
        annual_discount_rates = self.base_annual_discount_rate + annual_loss_rate
        monthly_discount_rates = annual_discount_rates / 12.0
        
        months_array = np.arange(1, self.num_months + 1, dtype=np.float32)
        
        # Broadcast monthly rates (num_accounts, 1) to months_array (99,) creating a (num_accounts, 99) discount matrix
        discount_factors = 1.0 / ((1.0 + monthly_discount_rates[:, None]) ** months_array)
        
        # Multiply cashflow matrix by discount factor matrix, then sum
        npv = np.sum(cashflows * discount_factors, axis=1)
        return npv

    def calculate_metrics(self, c1, c2, cashflows, npv):
        """
        Calculate additional business metrics efficiently across all accounts.
        Assumes c1 is PD (loss rate) and c2 is Balance.
        
        Calculates:
        - 5-yr Net Loss Rate
        - 5-yr Return on Asset (ROA)
        - 5-yr Return on Equity (ROE) - Assumes Equity is 10% of Balance
        - Payback Period (months)
        """
        # We look at 5 years (60 months), ensuring we don't exceed the num_months
        months_5yr = min(60, self.num_months)
        
        # Slices for first 5 years
        c1_5yr = c1[:, :months_5yr]
        c2_5yr = c2[:, :months_5yr]
        cf_5yr = cashflows[:, :months_5yr]
        
        # 5-yr Average Balance (Denominator for rates)
        avg_bal_5yr = np.mean(c2_5yr, axis=1)
        # Avoid division by zero
        safe_avg_bal = np.where(avg_bal_5yr == 0, 1e-9, avg_bal_5yr)
        
        # 1. 5-yr Net Loss Rate = Sum of Losses / Average Balance
        # Losses = PD (c1) * Balance (c2)
        losses_5yr = np.sum(c1_5yr * c2_5yr, axis=1)
        net_loss_rate_5yr = losses_5yr / safe_avg_bal
        
        # 2. 5-yr ROA = Sum of Cashflows / Average Balance
        total_cf_5yr = np.sum(cf_5yr, axis=1)
        roa_5yr = total_cf_5yr / safe_avg_bal
        
        # 3. 5-yr ROE = Sum of Cashflows / Average Equity
        # Assuming Equity = 10% of Average Balance
        avg_equity_5yr = safe_avg_bal * 0.10
        roe_5yr = total_cf_5yr / avg_equity_5yr
        
        # 4. Payback Period (months to positive cumulative cashflow)
        # Calculate cumulative sum of cashflows along the months axis
        cum_cf = np.cumsum(cashflows, axis=1)
        
        # Find the first month (index) where cumulative cashflow > 0
        # argmax returns the first index where condition is true.
        # We add 1 because month on book is 1-indexed (and indices are 0-indexed)
        positive_mask = cum_cf > 0
        payback_period = np.argmax(positive_mask, axis=1) + 1
        
        # If the account never pays back, it assigns a 0 payback. We can fix this by
        # explicitly setting those that never cross 0 to a default value like -1 or NaN.
        never_paid_back = ~np.any(positive_mask, axis=1)
        payback_period = np.where(never_paid_back, -1, payback_period)
        
        return net_loss_rate_5yr, roa_5yr, roe_5yr, payback_period

    def run(self, rdm_features, other_features):
        """
        End-to-end execution pipeline.
        Returns a Pandas DataFrame with account level metrics.
        """
        # Step 1: Initial RDM regression
        rdm = self.score_initial_rdm_model(rdm_features)
        
        # Step 2: 4 curve generation models using RDM
        c1, c2, c3, c4 = self.score_curve_models(rdm, other_features)
        
        # Step 3: Combine outputs to get 99-month cashflows
        cashflows = self.calculate_cashflows(c1, c2, c3, c4)
        
        # Step 4: Calculate Annual Loss Rate for Dynamic Discounting
        annual_loss_rate = self.calculate_annual_loss_rate(c1, c2)
        
        # Step 5: Discount for NPV
        npv = self.calculate_npv(cashflows, annual_loss_rate)
        
        # Step 5: Calculate Additional Metrics
        net_loss_rate_5yr, roa_5yr, roe_5yr, payback_period = self.calculate_metrics(c1, c2, cashflows, npv)
        
        # Bundle into a dataframe for the user
        results_df = pd.DataFrame({
            "NPV": npv,
            "5 Yr Net Loss Rate": net_loss_rate_5yr,
            "5 Yr ROA": roa_5yr,
            "5 Yr ROE": roe_5yr,
            "Payback Period (Months)": payback_period
        })
        
        return results_df


if __name__ == "__main__":
    NUM_ACCOUNTS = 1_000_000
    NUM_MONTHS = 99
    
    print(f"Generating synthetic characteristics for {NUM_ACCOUNTS:,} accounts...")
    
    # Generate 5 characteristics for the initial regression model (RDM)
    synthetic_rdm_features = np.random.randn(NUM_ACCOUNTS, 5).astype(np.float32)
    
    # Generate 3 additional characteristics for the downstream curve regression models
    synthetic_other_features = np.random.randn(NUM_ACCOUNTS, 3).astype(np.float32)
    
    # Initialize engine
    engine = AccountNPVEngine(num_months=NUM_MONTHS)
    
    print("Running Vectorized NPV Engine...")
    start_time = time.time()
    
    # Execute Pipeline
    results_df = engine.run(synthetic_rdm_features, synthetic_other_features)
    
    end_time = time.time()
    
    # Print statistics
    print("-" * 50)
    print(f"Engine completed in: {end_time - start_time:.4f} seconds.")
    print(f"Accounts Processed : {NUM_ACCOUNTS:,}")
    print(f"Months on book     : {NUM_MONTHS}")
    print(f"\nSample Account Results (First 5):")
    print(results_df.head(5).to_string())
    print("-" * 50)
