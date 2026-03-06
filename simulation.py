"""
Agent-based simulation of AI hallucinations enabling economic fraud.

This simulation implements the model described in:
"The Credibility Trap: How AI Hallucinations Enable Economic Fraud"
by Vanshika Pandey and Krishan Gupta.

Features:
- Heterogeneous users with trust levels drawn from a Beta distribution.
- Users can pay a verification cost to learn the true state.
- A strategic exploiter decides whether to launch a fraudulent scheme
  based on expected profit.
- Fraud occurs when a user invests in a bad state (S=0) after an AI hallucination
  and without verification.
"""

import numpy as np

# ---------------------------
# Core simulation function
# ---------------------------

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
    V=0.8,                   # exploiter's share of loss (per victim)
    F=50.0,                  # fixed cost to launch fraudulent scheme
    entry_decision_samples=2000  # samples to decide exploiter entry
):
    """
    Run one scenario of the agent-based model.

    Returns a dict with:
        mean_fraud_count : average number of fraud events per round
        mean_loss        : average aggregate loss per round
        mean_profit      : average exploiter profit per round (only when active)
        exploiter_active : whether the exploiter chose to enter
        fraud_probability: probability a random user falls victim in a round
    """
    # Compute investment threshold from payoffs
    threshold = L / (R + L)   # = 0.5 when R = L = 1

    # --- Step 1: Exploiter entry decision ---
    # Estimate the probability that a random user invests given S=0
    # by simulating a large number of independent users.
    def expected_victims_per_round():
        # Generate trust values from Beta distribution using method of moments
        var_trust = std_trust ** 2
        # Avoid division by zero or invalid parameters
        if var_trust >= mean_trust * (1 - mean_trust):
            # If variance too high, set to maximum possible for Beta
            k = 1e6  # effectively degenerate
        else:
            k = mean_trust * (1 - mean_trust) / var_trust - 1
        a = mean_trust * k
        b = (1 - mean_trust) * k
        # Draw trust and cost samples
        trust_samples = np.random.beta(a, b, size=entry_decision_samples)
        cost_samples = np.random.uniform(c_min, c_max, size=entry_decision_samples)

        # Simulate a round with S=0
        hallucination = np.random.random(entry_decision_samples) < H
        prior = 0.5
        # Posterior given A=1 (hallucination)
        posterior_given_A1 = prior + trust_samples * (1 - prior)

        # Would invest without verification?
        would_invest = (posterior_given_A1 > threshold) & hallucination
        # Verify if expected loss without verification > cost
        # Expected loss without verification = L if would_invest else 0
        verify = (would_invest * L) > cost_samples
        # Victim = hallucination & would_invest & not verify
        victim = hallucination & (posterior_given_A1 > threshold) & ~verify
        # Expected victims per round for N users
        return np.sum(victim) / entry_decision_samples * N

    expected_victims = expected_victims_per_round()
    expected_revenue = expected_victims * V
    exploiter_active = expected_revenue > F

    # --- Step 2: Main simulation (with exploiter if active) ---
    if not exploiter_active:
        return {
            'mean_fraud_count': 0.0,
            'mean_loss': 0.0,
            'mean_profit': 0.0,
            'exploiter_active': False,
            'fraud_probability': 0.0
        }

    # Precompute Beta parameters for user trust (same method as above)
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

    for _ in range(rounds):
        # State is always bad (S=0)
        # AI hallucinates this round with probability H
        hallucination_this_round = np.random.random() < H
        if not hallucination_this_round:
            # AI says A=0 (correct), so no user invests (posterior low)
            fraud_counts.append(0)
            losses.append(0)
            profits.append(0)
            continue

        # AI hallucinated: A=1 for all users in this round
        # Draw trust and cost for N users
        trust = np.random.beta(a, b, size=N)
        cost = np.random.uniform(c_min, c_max, size=N)

        # Posterior belief given A=1
        prior = 0.5
        posterior = prior + trust * (1 - prior)   # = 0.5*(1+trust)

        # Which users would invest without verification?
        would_invest = posterior > threshold

        # Which users verify? They verify if expected loss without verification > cost.
        # Expected loss = L if they would invest, else 0.
        verify = (would_invest * L) > cost

        # Victims: those who would invest AND do NOT verify
        victims = would_invest & ~verify
        fraud_count = np.sum(victims)
        fraud_counts.append(fraud_count)
        losses.append(fraud_count * L)
        profits.append(fraud_count * V)

    # Compute averages
    mean_fraud = np.mean(fraud_counts)
    mean_loss = np.mean(losses)
    mean_profit = np.mean(profits)
    fraud_prob = mean_fraud / N

    return {
        'mean_fraud_count': mean_fraud,
        'mean_loss': mean_loss,
        'mean_profit': mean_profit,
        'exploiter_active': True,
        'fraud_probability': fraud_prob
    }


# ---------------------------
# Main: run scenarios and print tables
# ---------------------------

def main():
    # Fixed parameters
    N = 1000
    rounds = 5000
    R = L = 1.0
    V = 0.8
    F = 50.0
    entry_samples = 2000
    trust_std = 0.15

    # Scenarios
    hallucination_rates = [0.1, 0.3, 0.6, 0.8]
    trust_levels = [0.3, 0.6, 0.8]          # mean trust

    # Verification cost ranges
    cost_baseline = (0.1, 0.5)
    cost_low = (0.05, 0.25)
    cost_high = (0.3, 1.0)

    print("=" * 70)
    print("AI HALLUCINATION AND ECONOMIC FRAUD: AGENT-BASED SIMULATION")
    print("=" * 70)
    print(f"Users per round: {N}, Rounds: {rounds}, R={R}, L={L}, V={V}, F={F}")
    print(f"Trust distribution: Beta with mean and std={trust_std}")
    print("-" * 70)

    # --- Table 1: Baseline (mean_trust=0.6, baseline costs) ---
    print("\nBASELINE SCENARIO: mean_trust = 0.6, verification cost = [0.1, 0.5]")
    print("-" * 70)
    print(f"{'H':<6} {'Exploiter active':<18} {'Fraud events':<15} {'Fraud prob':<12} {'Agg loss':<12} {'Exploiter profit':<18}")
    print("-" * 70)
    for H in hallucination_rates:
        res = simulate_scenario(
            H=H,
            mean_trust=0.6,
            std_trust=trust_std,
            c_min=cost_baseline[0],
            c_max=cost_baseline[1],
            N=N, rounds=rounds, R=R, L=L, V=V, F=F,
            entry_decision_samples=entry_samples
        )
        active = "Yes" if res['exploiter_active'] else "No"
        print(f"{H:<6.1f} {active:<18} {res['mean_fraud_count']:<15.2f} {res['fraud_probability']:<12.4f} {res['mean_loss']:<12.2f} {res['mean_profit']:<18.2f}")

    # --- Table 2: Trust sensitivity (H=0.3, baseline costs) ---
    print("\nTRUST SENSITIVITY (H=0.3, verification cost = [0.1,0.5])")
    print("-" * 70)
    print(f"{'Mean trust':<12} {'Exploiter active':<18} {'Fraud events':<15} {'Fraud prob':<12} {'Agg loss':<12}")
    print("-" * 70)
    for mu in trust_levels:
        res = simulate_scenario(
            H=0.3,
            mean_trust=mu,
            std_trust=trust_std,
            c_min=cost_baseline[0],
            c_max=cost_baseline[1],
            N=N, rounds=rounds, R=R, L=L, V=V, F=F,
            entry_decision_samples=entry_samples
        )
        active = "Yes" if res['exploiter_active'] else "No"
        print(f"{mu:<12.1f} {active:<18} {res['mean_fraud_count']:<15.2f} {res['fraud_probability']:<12.4f} {res['mean_loss']:<12.2f}")

    # --- Table 3: Verification cost sensitivity (H=0.3, mean_trust=0.6) ---
    print("\nVERIFICATION COST SENSITIVITY (H=0.3, mean_trust=0.6)")
    print("-" * 70)
    cost_scenarios = [("Low", cost_low), ("Baseline", cost_baseline), ("High", cost_high)]
    print(f"{'Cost level':<12} {'Exploiter active':<18} {'Fraud events':<15} {'Fraud prob':<12} {'Agg loss':<12}")
    print("-" * 70)
    for name, (cmin, cmax) in cost_scenarios:
        res = simulate_scenario(
            H=0.3,
            mean_trust=0.6,
            std_trust=trust_std,
            c_min=cmin,
            c_max=cmax,
            N=N, rounds=rounds, R=R, L=L, V=V, F=F,
            entry_decision_samples=entry_samples
        )
        active = "Yes" if res['exploiter_active'] else "No"
        print(f"{name:<12} {active:<18} {res['mean_fraud_count']:<15.2f} {res['fraud_probability']:<12.4f} {res['mean_loss']:<12.2f}")

    print("\n" + "=" * 70)
    print("Simulation complete.")


if __name__ == "__main__":
    main()
