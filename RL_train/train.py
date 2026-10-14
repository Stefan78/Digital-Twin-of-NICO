from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

# Connect to Unity Editor
print("Starting Python controller...")
print("Waiting for Unity to connect on port 5004...")

env = UnityEnvironment(base_port=5004, timeout_wait=120)

print("Unity connected!")
env.reset()

# Get behavior name from Unity Agent
behavior_name = list(env.behavior_specs.keys())[0]  # TODO: with multiple agents, it need not be 0

print("Behavior:", behavior_name)

while True:
    # Get current agents that need decisions
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Cycle for multiple agents, update those that we want
    # TODO: how do I find which agent corresponds to which ID?
    for agent_id in decision_steps.agent_id:
        obs = decision_steps[agent_id].obs[0]
        # TODO send observations to the NN predictor
        # TODO get the correct label somehow

        # Move randomly each step (currently using 11-dimensional actions)
        # TODO: replace with actual outputs from an NN
        random_move = (np.random.rand(11) - 0.5) * 2

        action = np.array(np.atleast_2d(random_move), dtype=np.float32)

        # Send action to this agent
        # TODO: not sure if this is gonna work, but we can try. Basically the "action" does not change anything,
        #  it just modifies the reward computation for the other agent (but that should be probably handled in C#
        #  "OnActionReceived" function)
        env.set_action_for_agent(
            behavior_name,
            agent_id,
            ActionTuple(continuous=action)
        )

    # Advance simulation
    env.step()
