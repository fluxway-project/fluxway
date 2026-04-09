import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import DPFeatures


@spmd(resource={"num_cpus": 1, "memory_gb": 1}, mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class DenseSPMD:
    """Final computation node: forward pass, loss, and dense/embedding gradient computation."""

    def __init__(
        self,
        embed_dim: int = 3,
        dense_stats_dim: int = 2,
        grad_accum_steps: int = 2,
        coordinator_address: str = "127.0.0.1:29500",
    ):
        self.embed_dim = embed_dim
        self.dense_stats_dim = dense_stats_dim
        self.grad_accum_steps = grad_accum_steps
        self._jax_initialized = False

        # self.rank and self.world_size are injected by the framework before __init__
        # is called, so jax.distributed.initialize is safe to call here.
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=self.world_size,
            process_id=self.rank,
        )

    def _ensure_jax_init(self):
        if self._jax_initialized:
            return

        n_global = len(jax.devices())
        print(
            f"[DenseSPMD._ensure_jax_init] rank={self.rank} "
            f"global_devices={n_global} local_devices={len(jax.local_devices())}"
        )

        self.mesh = Mesh(jax.devices(), ("dp",))

        input_dim = self.embed_dim * 2 + self.dense_stats_dim

        class DenseLayer(nnx.Module):
            def __init__(self, in_dim):
                self.linear = nnx.Linear(in_dim, 1, rngs=nnx.Rngs(0))

            def __call__(self, x):
                return self.linear(x).squeeze(-1)

        self.model = DenseLayer(input_dim)
        self.grad_accum = jtu.tree_map(
            lambda p: jnp.zeros_like(p), nnx.state(self.model)
        )
        self._jax_initialized = True

    def save_state(self) -> dict:
        self._ensure_jax_init()
        return {"model_state": nnx.state(self.model), "grad_accum": self.grad_accum}

    def restore_state(self, state: dict) -> None:
        self._ensure_jax_init()
        nnx.update(self.model, state["model_state"])
        self.grad_accum = state["grad_accum"]

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

        emb1_2d = emb1.flat_values.reshape(-1, emb1.flat_values.shape[-1])
        emb2_2d = emb2.flat_values.reshape(-1, emb2.flat_values.shape[-1])
        inv1_1d = inv1.flat_values.reshape(-1)
        inv2_1d = inv2.flat_values.reshape(-1)

        f1_rec = emb1_2d[inv1_1d]
        f2_rec = emb2_2d[inv2_1d]
        dense_stats = dp_data.dense_stats.flat_values.reshape(
            -1, dp_data.dense_stats.flat_values.shape[-1]
        )
        labels_local = dp_data.labels.is_click.flat_values.reshape(-1).astype(
            jnp.float32
        )

        dp_sharding = NamedSharding(self.mesh, P("dp"))
        replicated = NamedSharding(self.mesh, P())
        local_batch = f1_rec.shape[0]

        # Build global sharded arrays from per-rank local slices.
        # All 6 dp ranks collectively form the global batch of world_size * local_batch rows.
        global_f1 = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch, f1_rec.shape[1]),
            sharding=dp_sharding,
            arrays=[f1_rec],
        )
        global_f2 = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch, f2_rec.shape[1]),
            sharding=dp_sharding,
            arrays=[f2_rec],
        )
        global_stats = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch, dense_stats.shape[1]),
            sharding=dp_sharding,
            arrays=[dense_stats],
        )
        global_labels = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch,),
            sharding=dp_sharding,
            arrays=[labels_local],
        )

        @nnx.jit(
            in_shardings=(None, dp_sharding, dp_sharding, dp_sharding, dp_sharding),
            out_shardings=(None, dp_sharding, dp_sharding, replicated),
        )
        def forward_and_grad(model, f1, f2, stats, labels):
            x = jnp.concatenate([f1, f2, stats], axis=1)

            def loss_fn(m, input_x, y):
                pred = m(input_x)
                return jnp.mean((pred - y) ** 2)

            # dense_param_grads: replicated (same on all ranks after jit+allreduce)
            # x_grad: dp-sharded (each rank gets its local slice)
            dense_param_grads, x_grad = nnx.grad(loss_fn, argnums=(0, 1))(
                model, x, labels
            )
            loss_val = loss_fn(model, x, labels)
            emb1_dim = f1.shape[1]
            emb2_dim = f2.shape[1]
            return (
                dense_param_grads,
                x_grad[:, :emb1_dim],
                x_grad[:, emb1_dim : emb1_dim + emb2_dim],
                loss_val,
            )

        dense_param_grads, grad_f1_global, grad_f2_global, loss_val = forward_and_grad(
            self.model, global_f1, global_f2, global_stats, global_labels
        )

        # Extract this rank's local slice of the sharded x_grad using addressable_shards
        grad_f1_local = grad_f1_global.addressable_shards[0].data
        grad_f2_local = grad_f2_global.addressable_shards[0].data

        # scatter-add: (local_batch, emb_dim) → (K_u, emb_dim)
        g1_unique = jnp.zeros_like(emb1_2d).at[inv1_1d].add(grad_f1_local)
        g2_unique = jnp.zeros_like(emb2_2d).at[inv2_1d].add(grad_f2_local)

        dense_grad_leaves = jtu.tree_leaves(dense_param_grads)
        dense_grad_flat = jnp.concatenate([g.ravel() for g in dense_grad_leaves])

        print(
            f"[DenseSPMD.step] rank={self.rank} "
            f"f1_rec={f1_rec.shape} f2_rec={f2_rec.shape} "
            f"loss={float(loss_val):.4f} "
            f"dense_grad_norms={[float(jnp.linalg.norm(g)) for g in dense_grad_leaves]}"
        )

        return {
            "grad1": g1_unique,
            "grad2": g2_unique,
            "dense_grad": TransportTensor(
                dense_grad_flat, global_shape=(int(dense_grad_flat.shape[0]),)
            ),
        }

    @operator(
        name="update", input_sharding={"dense_grad": {"type": "static", "axis": []}}
    )
    def update(self, dense_grad: TransportTensor) -> None:
        step = self.ctx.step
        treedef = jtu.tree_structure(self.grad_accum)
        param_leaves = jtu.tree_leaves(nnx.state(self.model))
        flat = dense_grad.flat_values
        cursor = 0
        split_leaves = []
        for leaf in param_leaves:
            split_leaves.append(flat[cursor : cursor + leaf.size].reshape(leaf.shape))
            cursor += leaf.size

        self.grad_accum = jtu.tree_map(
            lambda acc, g: acc + g,
            self.grad_accum,
            jtu.tree_unflatten(treedef, split_leaves),
        )

        if step % self.grad_accum_steps == 0:
            params = nnx.state(self.model)
            updated = jtu.tree_map(lambda p, g: p - 0.01 * g, params, self.grad_accum)
            nnx.update(self.model, updated)
            self.grad_accum = jtu.tree_map(jnp.zeros_like, self.grad_accum)
            print(
                f"[DenseSPMD.update] rank={self.rank} step={step} updated model params"
            )
        else:
            print(f"[DenseSPMD.update] rank={self.rank} step={step} accumulating grads")
