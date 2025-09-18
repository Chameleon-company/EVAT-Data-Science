import json
import random

# Define action types and their relative probabilities
ACTION_MAP = [
    "check_in",
    "report_fault",
    "validate_ai_prediction",
    "discover_new_station_in_black_spot",
    "use_route_planner",
    "ask_chatbot_question"
]
ACTION_WEIGHTS = [10, 5, 3, 1, 5, 10]  # less common actions like "black_spot" have lower weights

def generate_random_actions(num_users=50, actions_per_user=70, seed=42, skip_chance=0.2):
    """
    Generates a list of simulated user actions with optional skipping logic.
    
    Args:
        num_users (int): Number of unique users to simulate.
        actions_per_user (int): Number of actions per active user.
        seed (int): Random seed for reproducibility.
        skip_chance (float): Probability that a user will not perform any action.

    Returns:
        list: List of simulated user action dictionaries.
    """
    random.seed(seed)
    simulated_data = []

    for user_id in range(1, num_users + 1):
        if random.random() < skip_chance:
            # User skips all actions
            continue

        for _ in range(actions_per_user):
            action_type = random.choices(ACTION_MAP, weights=ACTION_WEIGHTS, k=1)[0]
            simulated_data.append({
                "user_id": f"user_{user_id}",
                "action_type": action_type
                # "timestamp": random.randint(1700000000, 1800000000)  # Optional: UNIX timestamp
            })

    return simulated_data

# Save to JSON
if __name__ == "__main__":
    synthetic_actions = generate_random_actions()

    with open("simulated_user_actions.json", "w") as f:
        json.dump(synthetic_actions, f, indent=2)

    print(f"✅ Generated {len(synthetic_actions)} user actions across ~{len(set(x['user_id'] for x in synthetic_actions))} users.")
