from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.feature_space import feature_space
from cogrid.feature_space import feature
from cogrid.feature_space import features
from cogrid.envs.overcooked import overcooked_features
from cogrid import cogrid_env
from cogrid.core import grid_object
import numpy as np


class NAgentOvercookedFeatureSpace(feature.Feature):
    """
    A wrapper class to generate all encoded Overcooked features as a single array.

    For each agent j, calculate:

        - Agent j Direction
        - Agent j Inventory
        - Agent j Adjacent to Counter
        - Agent j Dist to closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
        - Agent j Pot Features for the two closest pots
            - pot_k_reachable: {0, 1}  # NOTE(chase): This is hardcoded to 1 currently.
            - pot_k_status: onehot of {empty | full | is_cooking | is_ready}
            - pot_k_contents: integer of the number of onions in the pot
            - pot_k_cooking_timer: integer for the number of ts remaining if cooking, 0 if finished, -1 if not cooking
            - pot_k_distance: (dy, dx) from the player's location
            - pot_k_location: (row, column) of the pot on the grid
        - Agent j Distance to other agents j != i
        - Agent j Position

    The observation is the concatenation of all these features for all players.
    """

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):

        num_agents = env.config["num_agents"]

        self.agent_features = [
            # Represent the direction of the agent
            features.AgentDir(),
            # The current inventory of the agent (max=1 item)
            overcooked_features.OvercookedInventory(),
            # One-hot indicator if there is a counter or pot in each of the four cardinal directions
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            # The (dy, dx) distance to the closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Onion, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Plate, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.PlateStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionSoup, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.DeliveryZone, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=grid_object.Counter, n=4
            ),
            # All pot features for the closest two pots
            overcooked_features.NClosestPotFeatures(num_pots=2),
            # The (dy, dx) distance to the closest other agent
            overcooked_features.DistToOtherPlayers(
                num_other_players=num_agents - 1
            ),
            # The (row, column) position of the agent
            features.AgentPosition(),
            # The direction the agent can move in
            features.CanMoveDirection(),
        ]

        full_shape = num_agents * np.sum(
            [feature.shape for feature in self.agent_features]
        )

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(
        self, env: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndarray:
        player_encodings = [self.generate_player_encoding(env, player_id)]

        for pid in env.agent_ids:
            if pid == player_id:
                continue
            player_encodings.append(self.generate_player_encoding(env, pid))

        encoding = np.hstack(player_encodings).astype(np.float32)

        assert np.array_equal(self.shape, encoding.shape)

        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)
