from typing import Optional, Sequence, Dict, Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory, VectorEncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction, _VectorEncoder
from d3rlpy.algos.iql import IQL
from d3rlpy.algos.torch.iql_impl import IQLImpl
from d3rlpy.algos.torch.ddpg_impl import DDPGBaseImpl
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler

from d3rlpy.models.torch.policies import NormalPolicy
from d3rlpy.models.torch.q_functions.mean_q_function import ContinuousMeanQFunction
from d3rlpy.models.torch import (
    ValueFunction,
)

from d3rlpy.models.builders import (
    create_non_squashed_normal_policy,
)

from d3rlpy.models.torch.imitators import DeterministicRegressor


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def fanin_init(tensor):
    """Initialize the weights of a layer with fan-in initialization.

    Args:
        tensor (torch.Tensor): Tensor to initialize.

    Returns:
        torch.Tensor: Initialized tensor.

    Raises:
        Exception: If the shape of the tensor is less than 2.
    """
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class CompositionalMlp(nn.Module):
    """Compositional MLP module."""

    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        num_modules: Sequence[int],
        module_assignment_positions: Sequence[int],
        module_inputs: Sequence[str],
        interface_depths: Sequence[int],
        graph_structure: Sequence[Sequence[int]],
        init_w: float = 3e-3,
        hidden_activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = nn.Identity,
        hidden_init: Optional[nn.Module] = fanin_init,
        b_init_value: float = 0.1,
        layer_norm: bool = False,
        layer_norm_kwargs: Optional[dict] = None,
    ):
        """Initialize the compositional MLP module.

        Args:
            sizes (list): List of sizes of each layer.
            num_modules (list): List of number of modules of each type.
            module_assignment_positions (list): List of module assignment positions.
            module_inputs (list): List of module inputs.
            interface_depths (list): List of interface depths.
            graph_structure (list): List of graph structures.
            init_w (float, optional): Initial weight value. Defaults to 3e-3.
            hidden_activation (nn.Module, optional): Hidden activation module. Defaults to nn.ReLU.
            output_activation (nn.Module, optional): Output activation module. Defaults to nn.Identity.
            hidden_init (function, optional): Hidden initialization function. Defaults to fanin_init.
            b_init_value (float, optional): Initial bias value. Defaults to 0.1.
            layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
            layer_norm_kwargs (dict, optional): Keyword arguments for layer normalization. Defaults to None.
        """
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.sizes = sizes
        self.num_modules = num_modules
        self.module_assignment_positions = module_assignment_positions
        self.module_inputs = module_inputs  # keys in a dict
        self.interface_depths = interface_depths
        self.graph_structure = (
            graph_structure  # [[0], [1,2], 3] or [[0], [1], [2], [3]]
        )
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.count = 0

        self.module_list = nn.ModuleList()  # e.g., object, robot, task...

        for graph_depth in range(
            len(graph_structure)
        ):  # root -> children -> ... leaves
            for j in graph_structure[
                graph_depth
            ]:  # loop over all module types at this depth
                self.module_list.append(nn.ModuleDict())  # pre, post
                self.module_list[j]["pre_interface"] = nn.ModuleList()
                self.module_list[j]["post_interface"] = nn.ModuleList()
                for k in range(num_modules[j]):  # loop over all modules of this type
                    layers_pre = []
                    layers_post = []
                    for i in range(
                        len(sizes[j]) - 1
                    ):  # loop over all depths in this module
                        if i == interface_depths[j]:
                            input_size = sum(
                                sizes[j_prev][-1]
                                for j_prev in graph_structure[graph_depth - 1]
                            )
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]

                        fc = nn.Linear(input_size, sizes[j][i + 1])
                        if (
                            graph_depth < len(graph_structure) - 1
                            or i < len(sizes[j]) - 2
                        ):
                            hidden_init(fc.weight)
                            fc.bias.data.fill_(b_init_value)
                            act = hidden_activation
                            layer_norm_this = layer_norm
                        else:
                            fc.weight.data.uniform_(-init_w, init_w)
                            fc.bias.data.uniform_(-init_w, init_w)
                            act = output_activation
                            layer_norm_this = None

                        if layer_norm_this is not None:
                            new_layer = [fc, nn.LayerNorm(sizes[j][i + 1]), act()]
                        else:
                            new_layer = [fc, act()]

                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer
                    if layers_pre:
                        self.module_list[j]["pre_interface"].append(
                            nn.Sequential(*layers_pre)
                        )
                    else:  # it's either a root or a module with no preprocessing
                        self.module_list[j]["pre_interface"].append(nn.Identity())
                    self.module_list[j]["post_interface"].append(
                        nn.Sequential(*layers_post)
                    )

    def forward(self, input_val: torch.Tensor, return_preactivations: bool = False):
        """Forward pass.

        Args:
            input_val (torch.Tensor): Input tensor.
            return_preactivations (bool, optional): Whether to return preactivations. Defaults to False.

        Returns:
            torch.Tensor: Output tensor.
        """
        if len(input_val.shape) > 2:
            input_val = input_val.squeeze(0)

        if return_preactivations:
            raise NotImplementedError("TODO: implement return preactivations")
        x = None
        for graph_depth in range(
            len(self.graph_structure)
        ):  # root -> children -> ... -> leaves
            x_post = []  # in case multiple module types at the same depth in the graph
            for j in self.graph_structure[graph_depth]:  # nodes (modules) at this depth
                if len(input_val.shape) == 1:
                    x_pre = input_val[self.module_inputs[j]]
                    onehot = input_val[self.module_assignment_positions[j]]
                    module_index = onehot.nonzero()[0]
                    x_pre = self.module_list[j]["pre_interface"][module_index](x_pre)
                    if x is not None:
                        x_pre = torch.cat((x, x_pre), dim=-1)
                    x_post.append(
                        self.module_list[j]["post_interface"][module_index](x_pre)
                    )
                else:
                    x_post_tmp = torch.empty(input_val.shape[0], self.sizes[j][-1]).to(
                        DEVICE
                    )
                    x_pre = input_val[:, self.module_inputs[j]]
                    onehot = input_val[:, self.module_assignment_positions[j]]
                    module_indices = onehot.nonzero(as_tuple=True)
                    assert (
                        module_indices[0]
                        == torch.arange(module_indices[0].shape[0]).to(DEVICE)
                    ).all()
                    module_indices_1 = module_indices[1]
                    for module_idx in range(self.num_modules[j]):
                        mask_inputs_for_this_module = module_indices_1 == module_idx
                        mask_to_input_idx = mask_inputs_for_this_module.nonzero()
                        x_pre_this_module = self.module_list[j]["pre_interface"][
                            module_idx
                        ](x_pre[mask_inputs_for_this_module])
                        if x is not None:
                            x_pre_this_module = torch.cat(
                                (x[mask_inputs_for_this_module], x_pre_this_module),
                                dim=-1,
                            )
                        x_post_this_module = self.module_list[j]["post_interface"][
                            module_idx
                        ](x_pre_this_module)
                        mask_to_input_idx = mask_to_input_idx.expand(
                            mask_to_input_idx.shape[0], x_post_this_module.shape[1]
                        )
                        x_post_tmp.scatter_(0, mask_to_input_idx, x_post_this_module)
                    x_post.append(x_post_tmp)
            x = torch.cat(x_post, dim=-1)
        return x


class _CompositionalEncoder(_VectorEncoder):  # type: ignore
    """_CompositionalEncoder class for d3rlpy."""

    def __init__(
        self,
        encoder_kwargs: dict,
        observation_shape: Sequence[int],
        init_w: float = 3e-3,
        *args,
        **kwargs,
    ):
        """Initialize _CompositionalEncoder class.

        Args:
            encoder_kwargs (dict): Encoder parameters.
            observation_shape (Sequence[int]): Observation shape.
            init_w (float, optional): Initial weight. Defaults to 3e-3.
        """
        super().__init__(
            observation_shape,
            hidden_units=None,
            use_batch_norm=False,
            dropout_rate=None,
            use_dense=False,
            activation=nn.ReLU(),
        )

        self._observation_shape = observation_shape
        self.encoder_kwargs = encoder_kwargs
        sizes = encoder_kwargs["sizes"]
        output_dim = encoder_kwargs["output_dim"]
        num_modules = encoder_kwargs["num_modules"]
        module_assignment_positions = encoder_kwargs["module_assignment_positions"]
        module_inputs = encoder_kwargs["module_inputs"]
        interface_depths = encoder_kwargs["interface_depths"]
        graph_structure = encoder_kwargs["graph_structure"]
        sizes = list(sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [output_dim]

        self._feature_size = sizes[-1][-1]

        self.comp_mlp = CompositionalMlp(
            sizes=sizes,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure,
            init_w=init_w,
        )

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp_mlp.forward(x)

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError("CompositionalEncoder does not have last_layer")


class CompositionalEncoder(_CompositionalEncoder, Encoder):
    """Implements the actual Compositional Encoder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simply runs the forward pass from _CompositionalEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self._fc_encode(x)


class CompositionalEncoderWithAction(_CompositionalEncoder, EncoderWithAction):
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, action], dim=1)
        h = self._fc_encode(x)
        return h


class CompositionalNonSquashedNormalPolicy(NormalPolicy):
    """CompositionalNonSquashedNormalPolicy class for d3rlpy."""

    def __init__(self, *args, **kwargs):
        """Initialize CompositionalNonSquashedNormalPolicy."""
        super().__init__(
            squash_distribution=False,
            *args,
            **kwargs,
        )
        self._mu = nn.Identity()


def create_non_squashed_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> CompositionalNonSquashedNormalPolicy:
    """Create a non-squashed normal policy.

    Args:
        observation_shape (Sequence[int]): Observation shape.
        action_size (int): Action size.
        encoder_factory (EncoderFactory): Encoder factory.
        min_logstd (float, optional): Minimum log standard deviation.
            Defaults to -20.0.
        max_logstd (float, optional): Maximum log standard deviation.
            Defaults to 2.0.
        use_std_parameter (bool, optional): Use std parameter. Defaults to False.

    Returns:
        CompositionalNonSquashedNormalPolicy: Non-squashed normal policy.
    """
    encoder = encoder_factory.create(observation_shape)
    return CompositionalNonSquashedNormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )


class CompositionalIQLImpl(IQLImpl, DDPGBaseImpl):
    """Compose IQL implementation class for d3rlpy."""

    _policy: Optional[CompositionalNonSquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        value_encoder_factory: EncoderFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        DDPGBaseImpl.__init__(
            self,
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=CompositionalMeanQFunctionFactory(),
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None

    def _build_actor(self) -> None:
        """Build actor network using the compositional encoder."""
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        super()._build_critic()
        self._value_func = create_compositional_value_function(
            self._observation_shape, self._value_encoder_factory
        )


def create_compositional_value_function(
    observation_shape: Sequence[int], encoder_factory: EncoderFactory
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    return CompositionalValueFunction(encoder)


class CompositionalValueFunction(ValueFunction):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder)
        self._fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)


class CompositionalIQL(IQL):
    """Compositional IQL class for d3rlpy."""

    def _create_impl(self, observation_shape: Sequence[int], action_size: int) -> None:
        """Create the implementation.

        Args:
            observation_shape (Sequence[int]): Observation shape.
            action_size (int): Action size.
        """
        self._impl = CompositionalIQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            value_encoder_factory=self._value_encoder_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            expectile=self._expectile,
            weight_temp=self._weight_temp,
            max_weight=self._max_weight,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()


class CompositionalContinuousMeanQFunction(ContinuousMeanQFunction):
    def __init__(self, encoder: EncoderWithAction):
        super().__init__(encoder)
        self._fc = nn.Identity()

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, action)


class CompositionalMeanQFunctionFactory(MeanQFunctionFactory):
    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ):
        raise NotImplementedError(
            "CompositionalMeanQFunctionFactory does not support discrete action spaces"
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> CompositionalContinuousMeanQFunction:
        return CompositionalContinuousMeanQFunction(encoder)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "share_encoder": self._share_encoder,
        }


class CompositionalEncoderFactory(VectorEncoderFactory):
    """Encoder factory for CompositionalEncoder."""

    def __init__(self, encoder_kwargs: dict, *args, **kwargs):
        """Initialize CompositionalEncoderFactory."""
        super().__init__(*args, **kwargs)
        self.encoder_kwargs = encoder_kwargs

    def create(self, observation_shape: Sequence[int]) -> CompositionalEncoder:
        """Create a CompositionalEncoder."""
        assert len(observation_shape) == 1
        return CompositionalEncoder(
            encoder_kwargs=self.encoder_kwargs,
            observation_shape=observation_shape,
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> CompositionalEncoderWithAction:
        return CompositionalEncoderWithAction(
            encoder_kwargs=self.encoder_kwargs,
            observation_shape=observation_shape,
            action_size=action_size,
            discrete_action=discrete_action,
        )


def create_cp_encoderfactory(with_action=False, output_dim=None):
    obs_dim = 93
    act_dim = 8
    # fmt: off
    observation_positions = {
        'object-state': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 
        'obstacle-state': np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 
        'goal-state': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]), 
        'object_id': np.array([45, 46, 47, 48]), 
        'robot_id': np.array([49, 50, 51, 52]), 
        'obstacle_id': np.array([53, 54, 55, 56]), 
        'subtask_id': np.array([57, 58, 59, 60]), 
        'robot0_proprio-state': np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92])}
    if with_action:
        observation_positions["action"] = np.array([93, 94, 95, 96, 97, 98, 99, 100])
    # fmt: on

    sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))
    module_names = ["obstacle_id", "object_id", "subtask_id", "robot_id"]
    module_input_names = [
        "obstacle-state",
        "object-state",
        "goal-state",
    ]
    if with_action:
        module_input_names.append(["robot0_proprio-state", "action"])
    else:
        module_input_names.append("robot0_proprio-state")

    module_assignment_positions = [observation_positions[key] for key in module_names]
    interface_depths = [-1, 1, 2, 3]
    graph_structure = [[0], [1], [2], [3]]
    num_modules = [len(onehot_pos) for onehot_pos in module_assignment_positions]

    module_inputs = []
    for key in module_input_names:
        if isinstance(key, list):
            # concatenate the inputs
            module_inputs.append(
                np.concatenate([observation_positions[k] for k in key], axis=0)
            )
        else:
            module_inputs.append(observation_positions[key])

    # module_inputs = [observation_positions[key]  for key in module_input_names]

    encoder_kwargs = {
        "sizes": sizes,
        "obs_dim": obs_dim,
        "output_dim": output_dim if output_dim is not None else act_dim,
        "num_modules": num_modules,
        "module_assignment_positions": module_assignment_positions,
        "module_inputs": module_inputs,
        "interface_depths": interface_depths,
        "graph_structure": graph_structure,
    }

    fac = CompositionalEncoderFactory(
        encoder_kwargs,
    )

    return fac

from d3rlpy.algos.bc import BC
from d3rlpy.models.torch.imitators import DeterministicRegressor
from d3rlpy.models.torch.policies import DeterministicPolicy, Policy
from d3rlpy.algos.torch.bc_impl import BCImpl
from d3rlpy.torch_utility import hard_sync

class CompositionalRegressor(DeterministicRegressor):
    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__(encoder, action_size)
        self._fc = nn.Identity()

class CompositionalBCPolicy(DeterministicPolicy):
    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__(encoder, action_size)
        self._fc = nn.Identity()

def create_comp_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: CompositionalEncoderFactory,
) -> CompositionalBCPolicy:
    encoder = encoder_factory.create(observation_shape)
    return CompositionalBCPolicy(encoder, action_size)

def create_comp_deterministic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: CompositionalEncoderFactory,
) -> CompositionalRegressor:
    encoder = encoder_factory.create(observation_shape)
    return CompositionalRegressor(encoder, action_size)

class CompositionalBC(BC):
    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CompositionalBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            policy_type=self._policy_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
        )
        self._impl.build()

class CompositionalBCImpl(BCImpl):
    def _build_network(self) -> None:
        if self._policy_type == "deterministic":
            self._imitator = create_comp_deterministic_regressor(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
            )
        elif self._policy_type == "stochastic":
            raise NotImplementedError("CompositionalBC does not support stochastic policies")
        else:
            raise ValueError("invalid policy_type: {self._policy_type}")

    @property
    def policy(self) -> Policy:
        assert self._imitator

        policy: Policy
        if self._policy_type == "deterministic":
            policy = create_comp_deterministic_policy(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
            )
        elif self._policy_type == "stochastic":
            raise NotImplementedError("CompositionalBC does not support stochastic policies")
        else:
            raise ValueError(f"invalid policy_type: {self._policy_type}")

        # copy parameters
        hard_sync(policy, self._imitator)

        return policy