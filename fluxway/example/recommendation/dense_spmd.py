import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils

from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import DPFeatures


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class DenseSPMD:
    """Final computation node: forward pass, loss, and dense/embedding gradient computation."""

    def __init__(self, dense_dim: int = 1, grad_accum_steps: int = 2):
        self.dense_dim = dense_dim
        self.grad_accum_steps = grad_accum_steps
        self.grad_accum = jnp.zeros((dense_dim,), dtype=jnp.float32)
        self._jax_initialized = False

    def _ensure_jax_init(self):
        if self._jax_initialized:
            return

        try:
            jax.distributed.initialize(
                coordinator_address="127.0.0.1:23456",
                num_processes=self.world_size,
                process_id=self.rank,
            )
        except RuntimeError:
            pass

        self.mesh = Mesh(jax.devices(), ("dp",))

        class DenseLayer(nnx.Module):
            def __init__(self, dim):
                self.bias = nnx.Param(jnp.array([1.0], dtype=jnp.float32))

            def __call__(self, x):
                return self.bias.value * x

        self.model = DenseLayer(self.dense_dim)
        self._jax_initialized = True

    @operator(
        name="step",
        input_sharding={
            "emb1": {"type": "static", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "static", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "grad1": {"type": "mirror", "reference": "step.emb1"},
            "grad2": {"type": "mirror", "reference": "step.emb2"},
            "dense_grad": {"type": "static", "axis": []},
        },
    )
    def step(
        self,
        emb1: TransportTensor,
        inv1: TransportTensor,
        emb2: TransportTensor,
        inv2: TransportTensor,
        dp_data: DPFeatures,
    ) -> dict:
        self._ensure_jax_init()

        f1_rec = emb1[inv1]
        f2_rec = emb2[inv2]

        local_x = f1_rec.sum() + f2_rec.sum() + dp_data.dense_stats.sum()

        @nnx.jit(
            in_shardings=(None, NamedSharding(self.mesh, P("dp"))), out_shardings=None
        )
        def compute_dense_grad(model, x):
            def loss_fn(m, input_x):
                pred = m(input_x)
                return jnp.mean((pred - 5.0) ** 2)

            return nnx.grad(loss_fn)(model, x)

        with self.mesh:
            local_data = jnp.array([local_x], dtype=jnp.float32)
            global_data = multihost_utils.host_local_array_to_global_array(
                local_data, self.mesh, P("dp")
            )
            grad_tree = compute_dense_grad(self.model, global_data)
            global_dense_grad = jtu.tree_leaves(grad_tree)[0][0]

        f1_shape = (len(inv1), emb1.flat_values.shape[1])
        f2_shape = (len(inv2), emb2.flat_values.shape[1])

        g1_gathered = jnp.ones(f1_shape, dtype=jnp.float32) * 0.1
        g2_gathered = jnp.ones(f2_shape, dtype=jnp.float32) * 0.2

        g1_unique = (
            jnp.zeros((len(emb1), f1_shape[1]))
            .at[inv1.flat_values.reshape(-1)]
            .add(g1_gathered)
        )
        g2_unique = (
            jnp.zeros((len(emb2), f2_shape[1]))
            .at[inv2.flat_values.reshape(-1)]
            .add(g2_gathered)
        )

        print(
            f"[DenseSPMD.step] rank={self.rank} "
            f"user_emb={emb1.flat_values} item_emb={emb2.flat_values} "
            f"dense_grad={global_dense_grad}"
        )

        return {
            "grad1": g1_unique,
            "grad2": g2_unique,
            "dense_grad": TransportTensor(global_dense_grad, global_shape=()),
        }

    @operator(
        name="update", input_sharding={"dense_grad": {"type": "static", "axis": []}}
    )
    def update(self, dense_grad: jnp.ndarray) -> None:
        step = self.ctx.step
        self.grad_accum = self.grad_accum + dense_grad

        if step % self.grad_accum_steps == 0:
            self.model.bias.value = self.model.bias.value - 0.01 * self.grad_accum
            self.grad_accum = jnp.zeros_like(self.grad_accum)
            print(
                f"[DenseSPMD.update] rank={self.rank} step={step} "
                f"bias={self.model.bias.value}"
            )
        else:
            print(
                f"[DenseSPMD.update] rank={self.rank} step={step} "
                f"accumulating grad_accum={self.grad_accum}"
            )
