"""
Agent-based simulation of AI hallucinations enabling economic fraud.
Empirically calibrated using:
- Hallucination rates (H) from DefAn dataset (UWA, 2025)
- Trust parameters (T) from KPMG Global AI Trust Study 2025
- Loss magnitudes (L) from FCA APP Fraud Dataset
"""

import numpy as np
from scipy import stats   # for confidence intervals

def simulate_scenario(
    H,                       # hallucination rate
    mean_trust,              # mean of user trust distribution
    std_trust,               # standard deviation of trust distribution
    c_min,                   # minimum verification cost
    c_max,                   # maximum verification cost
    N=1000,                  # number of users per round
    rounds=5000,             # number of simulation rounds
    R=1.0,                   # gain if investment succeeds (state good)
    L=1.0,                   # loss if investment fails (state bad)
    V=0.9,                   # exploiter's share of loss (per victim)
    F=50.0,                  # fixed cost to launch fraudulent scheme
    entry_decision_samples=2000  # samples to decide exploiter entry
):
    """
    Run one scenario of the agent-based model.

    Returns a dict with:
        mean_fraud_count : average number of fraud events per round
        std_fraud_count  : standard deviation of fraud events per round
        ci_low, ci_high  : 95% confidence interval for fraud count
        mean_loss        : average aggregate loss per round
        mean_profit      : average exploiter profit per round (only when active)
        mean_verif_cost  : average verification cost per round
        exploiter_active : whether the exploiter chose to enter
        fraud_probability: probability a random user falls victim in a round
    """
    threshold = L / (R + L)

    def expected_victims_per_round():
        var_trust = std_trust ** 2
        if var_trust >= mean_trust * (1 - mean_trust):
            k = 1e6
        else:
            k = mean_trust * (1 - mean_trust) / var_trust - 1
        a = mean_trust * k
        b = (1 - mean_trust) * k
        trust_samples = np.random.beta(a, b, size=entry_decision_samples)
        cost_samples = np.random.uniform(c_min, c_max, size=entry_decision_samples)

        hallucination = np.random.random(entry_decision_samples) < H
        prior = 0.5
        posterior_given_A1 = prior + trust_samples * (1 - prior)

        would_invest = (posterior_given_A1 > threshold) & hallucination
        verify = (would_invest * L) > cost_samples
        victim = hallucination & (posterior_given_A1 > threshold) & ~verify
        return np.sum(victim) / entry_decision_samples * N

    expected_victims = expected_victims_per_round()
    expected_revenue = expected_victims * V
    exploiter_active = expected_revenue > F

    if not exploiter_active:
        return {
            'mean_fraud_count': 0.0,
            'std_fraud_count': 0.0,
            'ci_low': 0.0,
            'ci_high': 0.0,
            'mean_loss': 0.0,
            'mean_profit': 0.0,
            'mean_verif_cost': 0.0,
            'exploiter_active': False,
            'fraud_probability': 0.0
        }

    var_trust = std_trust ** 2
    if var_trust >= mean_trust * (1 - mean_trust):
        k = 1e6
    else:
        k = mean_trust * (1 - mean_trust) / var_trust - 1
    a = mean_trust * k
    b = (1 - mean_trust) * k

    fraud_counts = []
    losses = []
    profits = []
    verification_costs = []

    for _ in range(rounds):
        hallucination_this_round = np.random.random() < H
        if not hallucination_this_round:
            fraud_counts.append(0)
            losses.append(0)
            profits.append(0)
            verification_costs.append(0.0)
            continue

        trust = np.random.beta(a, b, size=N)
        cost = np.random.uniform(c_min, c_max, size=N)

        prior = 0.5
        posterior = prior + trust * (1 - prior)

        would_invest = posterior > threshold
        verify = (would_invest * L) > cost
        victims = would_invest & ~verify
        fraud_count = np.sum(victims)
        fraud_counts.append(fraud_count)
        losses.append(fraud_count * L)
        profits.append(fraud_count * V)
        # Sum verification costs for users who verified
        verif_cost_round = np.sum(cost[verify])
        verification_costs.append(verif_cost_round)

    mean_fraud = np.mean(fraud_counts)
    std_fraud = np.std(fraud_counts, ddof=1)   # sample standard deviation
    n = len(fraud_counts)
    sem = stats.sem(fraud_counts)              # standard error of the mean
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_fraud, scale=sem)

    mean_loss = np.mean(losses)
    mean_profit = np.mean(profits)
    mean_verif_cost = np.mean(verification_costs)
    fraud_prob = mean_fraud / N

    return {
        'mean_fraud_count': mean_fraud,
        'std_fraud_count': std_fraud,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'mean_loss': mean_loss,
        'mean_profit': mean_profit,
        'mean_verif_cost': mean_verif_cost,
        'exploiter_active': True,
        'fraud_probability': fraud_prob
    }

def main():
    # ============================================================
    # EMPIRICAL CALIBRATION (based on real-world datasets)
    # ============================================================
    # Hallucination rates from DefAn dataset (Rahman et al., 2025)
    H_empirical = [0.31, 0.55, 0.82]

    # Trust levels from KPMG Global AI Trust Study 2025 (Gillespie & Lockey, 2025)
    T_empirical = [0.40, 0.46, 0.68]

    # Loss magnitude from FCA APP Fraud Dataset (Financial Conduct Authority, 2023)
    L_empirical = 1000.0   # GBP

    V_empirical = 0.9
    F_empirical = 50.0
    N_empirical = 1000
    rounds_empirical = 5000
    trust_std_empirical = 0.15
    R_empirical = L_empirical   # gain equals loss

    print("=" * 80)
    print("AI HALLUCINATION AND ECONOMIC FRAUD: EMPIRICALLY CALIBRATED SIMULATION")
    print("=" * 80)
    print(f"Users per round: {N_empirical}, Rounds: {rounds_empirical}")
    print(f"R = L = £{L_empirical}, V = {V_empirical}, F = £{F_empirical}")
    print(f"Trust distribution: Beta with mean and std={trust_std_empirical}")
    print("-" * 80)

    # --- Baseline: medium trust (global average 0.46), varying H ---
    print("\nBASELINE SCENARIO: Trust = 0.46 (global average), varying hallucination rate H")
    print(f"{'H':<8} {'Active':<8} {'Fraud Events (Mean ± SD)':<28} {'[95% CI]':<20} {'Fraud prob':<12} {'Agg loss (£)':<15} {'Expl profit (£)':<15} {'Verif cost (£)':<15}")
    print("-" * 110)
    for H in H_empirical:
        res = simulate_scenario(
            H=H, mean_trust=0.46, std_trust=trust_std_empirical,
            c_min=0.1, c_max=0.5, N=N_empirical, rounds=rounds_empirical,
            R=R_empirical, L=L_empirical, V=V_empirical, F=F_empirical,
            entry_decision_samples=2000
        )
        active = "Yes" if res['exploiter_active'] else "No"
        mean_f = res['mean_fraud_count']
        std_f = res['std_fraud_count']
        ci_low = res['ci_low']
        ci_high = res['ci_high']
        print(f"{H:<8.2f} {active:<8} {mean_f:<15.2f} ± {std_f:<10.2f}       [{ci_low:.2f}, {ci_high:.2f}] {res['fraud_probability']:<12.4f} {res['mean_loss']:<15.2f} {res['mean_profit']:<15.2f} {res['mean_verif_cost']:<15.2f}")

    # --- Trust sensitivity: medium hallucination H=0.55, varying T ---
    print("\nTRUST SENSITIVITY: H = 0.55 (medium hallucination), varying trust T")
    print(f"{'Mean trust':<12} {'Active':<8} {'Fraud Events (Mean ± SD)':<28} {'[95% CI]':<20} {'Fraud prob':<12} {'Agg loss (£)':<15} {'Verif cost (£)':<15}")
    print("-" * 100)
    for mu in T_empirical:
        res = simulate_scenario(
            H=0.55, mean_trust=mu, std_trust=trust_std_empirical,
            c_min=0.1, c_max=0.5, N=N_empirical, rounds=rounds_empirical,
            R=R_empirical, L=L_empirical, V=V_empirical, F=F_empirical,
            entry_decision_samples=2000
        )
        active = "Yes" if res['exploiter_active'] else "No"
        mean_f = res['mean_fraud_count']
        std_f = res['std_fraud_count']
        ci_low = res['ci_low']
        ci_high = res['ci_high']
        print(f"{mu:<12.2f} {active:<8} {mean_f:<15.2f} ± {std_f:<10.2f}       [{ci_low:.2f}, {ci_high:.2f}] {res['fraud_probability']:<12.4f} {res['mean_loss']:<15.2f} {res['mean_verif_cost']:<15.2f}")

    # --- Verification cost sensitivity: medium H=0.55, medium T=0.46 ---
    print("\nVERIFICATION COST SENSITIVITY: H = 0.55, trust = 0.46")
    print(f"{'Cost scenario':<15} {'Active':<8} {'Fraud Events (Mean ± SD)':<28} {'[95% CI]':<20} {'Fraud prob':<12} {'Agg loss (£)':<15} {'Verif cost (£)':<15}")
    print("-" * 100)

    cost_ranges = [
        ("Low (0.05-0.25)", 0.05, 0.25),
        ("Baseline (0.1-0.5)", 0.1, 0.5),
        ("High (0.3-1.0)", 0.3, 1.0)
    ]
    for name, cmin, cmax in cost_ranges:
        res = simulate_scenario(
            H=0.55, mean_trust=0.46, std_trust=trust_std_empirical,
            c_min=cmin, c_max=cmax, N=N_empirical, rounds=rounds_empirical,
            R=R_empirical, L=L_empirical, V=V_empirical, F=F_empirical,
            entry_decision_samples=2000
        )
        active = "Yes" if res['exploiter_active'] else "No"
        mean_f = res['mean_fraud_count']
        std_f = res['std_fraud_count']
        ci_low = res['ci_low']
        ci_high = res['ci_high']
        print(f"{name:<15} {active:<8} {mean_f:<15.2f} ± {std_f:<10.2f}       [{ci_low:.2f}, {ci_high:.2f}] {res['fraud_probability']:<12.4f} {res['mean_loss']:<15.2f} {res['mean_verif_cost']:<15.2f}")

    # --- Exploiter parameter sensitivity: vary F and V ---
    print("\nEXPLOITER PARAMETER SENSITIVITY: H = 0.55, Trust = 0.46")
    print(f"{'F (£)':<8} {'V':<6} {'Active':<8} {'Fraud Events (Mean ± SD)':<28} {'[95% CI]':<20} {'Agg loss (£)':<15} {'Expl profit (£)':<15} {'Verif cost (£)':<15}")
    print("-" * 120)

    F_values = [10, 50, 200, 500]
    V_values = [0.5, 0.7, 0.9]

    for F_val in F_values:
        for V_val in V_values:
            res = simulate_scenario(
                H=0.55, mean_trust=0.46, std_trust=trust_std_empirical,
                c_min=0.1, c_max=0.5, N=N_empirical, rounds=rounds_empirical,
                R=R_empirical, L=L_empirical, V=V_val, F=F_val,
                entry_decision_samples=2000
            )
            active = "Yes" if res['exploiter_active'] else "No"
            mean_f = res['mean_fraud_count']
            std_f = res['std_fraud_count']
            ci_low = res['ci_low']
            ci_high = res['ci_high']
            print(f"{F_val:<8} {V_val:<6} {active:<8} {mean_f:<15.2f} ± {std_f:<10.2f}       [{ci_low:.2f}, {ci_high:.2f}] {res['mean_loss']:<15.2f} {res['mean_profit']:<15.2f} {res['mean_verif_cost']:<15.2f}")

    print("\n" + "=" * 80)
    print("Simulation complete.")

if __name__ == "__main__":
    main()
