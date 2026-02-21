import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

class ICAAP_Model_Consumer:
    """
    A Pillar 2A Stress Test Model tailored for a CONSUMER Loan Portfolio under Basel 3.1.
    
    Key distinctions for Consumer Portfolio:
    - Products: Mortgages, Credit Cards, Auto Loans, Personal Loans.
    - Pillar 1 (SA): Risk weights driven by Product Type and Loan-to-Value (LTV) for Mortgages.
    - Pillar 2A: 
        - Concentration Risk (Product/Geographic).
        - Conduct Risk (High relevance for consumer banks).
    """

    def __init__(self, portfolio_size=5000):
        self.portfolio_size = portfolio_size
        self.portfolio = self._generate_consumer_portfolio()
        
        # Base RWA components (Dummy values)
        self.market_risk_rwa = 10_000_000  # Lower for consumer banks usually
        self.op_risk_rwa = 80_000_000      # Higher due to high transaction volume/fraud risk

    def _generate_consumer_portfolio(self):
        """Generates a dummy consumer loan portfolio with specific product characteristics."""
        ids = range(self.portfolio_size)
        
        # Define Consumer Products and probability
        products = ['Mortgage', 'Credit Card', 'Auto Loan', 'Personal Loan']
        prod_probs = [0.40, 0.30, 0.20, 0.10] # 40% Mortgages
        
        assigned_products = np.random.choice(products, self.portfolio_size, p=prod_probs)
        
        # Generate Exposures based on Product Type
        exposures = []
        ltvs = []
        
        for p in assigned_products:
            if p == 'Mortgage':
                exp = np.random.lognormal(mean=12.2, sigma=0.4) # ~200k avg
                ltv = np.random.uniform(0.40, 0.95)             # 40% to 95% LTV
            elif p == 'Auto Loan':
                exp = np.random.lognormal(mean=9.9, sigma=0.3)  # ~20k avg
                ltv = np.random.uniform(0.60, 1.00)             # Secured but depreciating
            elif p == 'Credit Card':
                exp = np.random.lognormal(mean=7.6, sigma=0.8)  # ~2k avg
                ltv = 0.0 # Unsecured
            else: # Personal Loan
                exp = np.random.lognormal(mean=9.2, sigma=0.5)  # ~10k avg
                ltv = 0.0 # Unsecured
            
            exposures.append(exp)
            ltvs.append(ltv)
            
        df = pd.DataFrame({
            'LoanID': ids,
            'Product': assigned_products,
            'Exposure': exposures,
            'LTV': ltvs, # Loan-to-Value Ratio (Critical for Basel 3.1 Mortgages)
            'CreditScore': np.random.randint(580, 850, self.portfolio_size), # FICO-like
            'PD': np.random.uniform(0.001, 0.08, self.portfolio_size), # Prob of Default
            'LGD': np.random.uniform(0.10, 0.45, self.portfolio_size)  # Loss Given Default (Lower for secured)
        })
        return df

    def _get_basel_3_1_risk_weight(self, row):
        """
        Assigns Standardised Approach (SA) Risk Weights based on Basel 3.1 rules.
        """
        product = row['Product']
        ltv = row['LTV']
        
        if product == 'Mortgage':
            # Basel 3.1 Residential Real Estate RW (Simplified SCRA)
            if ltv <= 0.50: return 0.20
            elif ltv <= 0.60: return 0.25
            elif ltv <= 0.80: return 0.30
            elif ltv <= 0.90: return 0.40
            elif ltv <= 1.00: return 0.50
            else: return 0.70 # High LTV
            
        elif product == 'Credit Card':
            # Regulatory Retail (Qualifying Revolving) - typically 75%
            return 0.75
            
        elif product == 'Auto Loan':
            # Regulatory Retail - 75%
            return 0.75
            
        elif product == 'Personal Loan':
            # Other Retail - often 75% if diversified, else 100%
            # Using 100% for higher risk unsecured personal loans for conservatism
            return 1.00
            
        return 1.00 # Fallback

    def calculate_pillar_1_capital(self):
        """Calculates Pillar 1 Capital."""
        # Apply Risk Weights
        self.portfolio['RiskWeight'] = self.portfolio.apply(self._get_basel_3_1_risk_weight, axis=1)
        self.portfolio['RWA'] = self.portfolio['Exposure'] * self.portfolio['RiskWeight']
        
        credit_rwa = self.portfolio['RWA'].sum()
        total_rwa = credit_rwa + self.market_risk_rwa + self.op_risk_rwa
        capital_req = total_rwa * 0.08
        
        return total_rwa, capital_req

    def calculate_pillar_2a_concentration(self):
        """
        Pillar 2A: Product Concentration Risk.
        (Focuses on heavy reliance on one product type, e.g., only Mortgages).
        """
        product_exposure = self.portfolio.groupby('Product')['Exposure'].sum()
        total_exposure = product_exposure.sum()
        
        # HHI Index
        shares = product_exposure / total_exposure
        hhi = (shares ** 2).sum()
        
        # Threshold: if HHI > 0.40 (Consumer banks are often specialized, so higher threshold)
        if hhi > 0.40:
            add_on = (hhi - 0.40) * total_exposure * 0.05
        else:
            add_on = 0.0
            
        return add_on, hhi

    def calculate_pillar_2a_conduct_risk(self):
        """
        Pillar 2A: Conduct Risk (Specific to Consumer).
        Risk of fines/redress costs due to mis-selling (e.g., PPI, hidden fees).
        """
        # Heuristic: 0.5% of total unsecured exposure (Higher risk of conduct issues in unsecured)
        unsecured_exposure = self.portfolio[self.portfolio['Product'].isin(['Credit Card', 'Personal Loan'])]['Exposure'].sum()
        conduct_add_on = unsecured_exposure * 0.005
        return conduct_add_on

    def calculate_pillar_2a_irrbb(self):
        """
        Pillar 2A: IRRBB for Consumer.
        Key Factor: Prepayment Risk on Mortgages when rates fall.
        """
        # 1. Standard Value Calculation (Duration Gap)
        # Consumer banks often have longer asset duration (Fixed Rate Mortgages) vs short liabilities (Deposits)
        duration_gap = 3.0 # Years
        rate_shock = 0.02
        total_assets = self.portfolio['Exposure'].sum() * 1.2 # Approx
        
        eve_impact = duration_gap * rate_shock * total_assets
        
        # 2. Add Prepayment Risk Component
        # If rates drop, mortgage customers refinance -> Asset duration shortens -> Reinvestment risk
        mortgage_exposure = self.portfolio[self.portfolio['Product'] == 'Mortgage']['Exposure'].sum()
        prepayment_risk_add_on = mortgage_exposure * 0.002 # 20bps add-on estimate
        
        total_irrbb = (eve_impact * 0.5) + prepayment_risk_add_on
        return total_irrbb

    def run_stress_test(self, scenario='Base'):
        """
        Runs stress test including House Price Shocks impacting LTVs.
        """
        print(f"--- Running Stress Test: {scenario} ---")
        
        # Scenarios Definition
        # HPI: House Price Index change (Base: 0%, Severe: -25%)
        scenarios = {
            'Base':   {'PD_mult': 1.0, 'HPI_change': 0.00,  'Conduct_mult': 1.0},
            'Mild':   {'PD_mult': 1.3, 'HPI_change': -0.10, 'Conduct_mult': 1.1},
            'Severe': {'PD_mult': 2.5, 'HPI_change': -0.25, 'Conduct_mult': 1.5}
        }
        params = scenarios.get(scenario, scenarios['Base'])
        
        stressed_port = self.portfolio.copy()
        
        # 1. Apply House Price Shock to LTVs
        # New LTV = Loan / (Value * (1 + HPI_change))
        # Approximation: New LTV = Old LTV / (1 + HPI_change)
        stressed_port['LTV'] = stressed_port['LTV'] / (1 + params['HPI_change'])
        
        # 2. Recalculate Risk Weights (Migrate LTV bands)
        # Higher LTVs -> Higher RWs for Mortgages
        stressed_port['RiskWeight'] = stressed_port.apply(self._get_basel_3_1_risk_weight, axis=1)
        
        # 3. Calculate Stressed RWA (Pillar 1)
        stressed_port['RWA'] = stressed_port['Exposure'] * stressed_port['RiskWeight']
        credit_rwa = stressed_port['RWA'].sum()
        
        # Market/Op Risk Stress
        total_rwa = credit_rwa + self.market_risk_rwa + (self.op_risk_rwa * params.get('OpRisk_mult', 1.0))
        p1_capital = total_rwa * 0.08
        
        # 4. Stressed Pillar 2A
        p2a_conc, _ = self.calculate_pillar_2a_concentration()
        
        # Conduct Risk often correlates with economic downturns (more complaints)
        p2a_conduct = self.calculate_pillar_2a_conduct_risk() * params['Conduct_mult']
        
        # IRRBB stays similar or increases
        p2a_irrbb = self.calculate_pillar_2a_irrbb()
        
        total_p2a = p2a_conc + p2a_conduct + p2a_irrbb
        total_required = p1_capital + total_p2a
        
        print(f"Total Exposure: ${stressed_port['Exposure'].sum():,.2f}")
        print(f"Total RWA: ${total_rwa:,.2f}")
        print(f"Pillar 1 Capital: ${p1_capital:,.2f}")
        print(f"Pillar 2A (Conduct): ${p2a_conduct:,.2f}")
        print(f"Pillar 2A (IRRBB): ${p2a_irrbb:,.2f}")
        print(f"TOTAL CAPITAL REQ: ${total_required:,.2f}")
        print("-" * 30)
        
        return {
            'Scenario': scenario,
            'Total_RWA': total_rwa,
            'Capital_Req': total_required
        }

if __name__ == "__main__":
    model = ICAAP_Model_Consumer(portfolio_size=2000)
    
    results = []
    results.append(model.run_stress_test('Base'))
    results.append(model.run_stress_test('Severe'))
    
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(pd.DataFrame(results))
