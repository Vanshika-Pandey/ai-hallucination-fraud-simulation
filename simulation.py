import random

def run_simulation(hallucination_rate, trust_rate, avg_loss, iterations):
    fraud_count = 0

    for _ in range(iterations):
        hallucination = random.random() < hallucination_rate
        trust = random.random() < trust_rate

        if hallucination and trust:
            fraud_count += 1

    fraud_probability = fraud_count / iterations
    total_loss = fraud_count * avg_loss

    return fraud_count, fraud_probability, total_loss


def main():
    # Fixed Parameters
    trust_rate = 0.6          # 60% user trust
    avg_loss = 1000           # Average financial loss per fraud event
    iterations = 10000        # Number of simulation cycles

    # Different hallucination scenarios
    hallucination_scenarios = [0.1, 0.3, 0.6]

    print("AI Hallucination - Fraud Simulation Model")
    print("Iterations per Scenario:", iterations)
    print("Trust Rate:", trust_rate)
    print("Average Loss per Fraud:", avg_loss)
    print("-" * 50)

    for h_rate in hallucination_scenarios:
        fraud_count, fraud_probability, total_loss = run_simulation(
            h_rate, trust_rate, avg_loss, iterations
        )

        print(f"Hallucination Rate: {h_rate}")
        print(f"Fraud Occurrences: {fraud_count}")
        print(f"Fraud Probability: {round(fraud_probability, 4)}")
        print(f"Total Economic Loss: {total_loss}")
        print("-" * 50)


if __name__ == "__main__":
    main()
