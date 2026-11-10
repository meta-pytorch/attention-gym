# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ruff: noqa
# Vendored test oracle for attn_gym.warpdrive.examples.external_maps.CutlassRaster.
# Source: https://github.com/NVIDIA/cutlass, examples/python/CuTeDSL/helpers/
# static_persistent_tile_scheduler.py at e13f5cd4142c43efc37c2f6339923eb4a57afb99 (the BSD-3
# replacement of the deprecated cutlass.utils copy; not shipped in the nvidia-cutlass-dsl wheel).
# Changes: StaticPersistentRuntimeTileScheduler, the host-side get_grid_shape helpers, the
# num_tiles_executed property and the Sphinx signature shim omitted (unused by the oracle);
# reformatted by pre-commit; comments and docstrings shortened.

from cutlass.cutlass_dsl import (
    Boolean,
    Integer,
    Int32,
    extract_mlir_values,
    new_from_mlir_values,
    dsl_user_op,
    const_expr,
)
from cutlass._mlir import ir
import cutlass.cute as cute

# Static persistent tile scheduler


class WorkTileInfo:
    """A scheduled tile coordinate and its validity flag."""

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.tile_idx)
        values.extend(extract_mlir_values(self.is_valid_tile))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 4
        new_tile_idx = new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    @cute.jit
    def is_valid_tile(self) -> Boolean:
        """Return false for scheduling requests after all work is complete."""
        return self._is_valid_tile

    @property
    @cute.jit
    def tile_idx(self) -> cute.Coord:
        return self._tile_idx


class PersistentTileSchedulerParams:
    """Cluster and tile layouts for a batched GEMM schedule.

    ``cluster_shape_mn`` counts CTAs along M and N; the K extent must be 1.
    ``problem_layout_ncluster_mnl`` counts clusters along M, N, and batch.
    """

    @dsl_user_op
    def __init__(
        self,
        problem_shape_ntile_mnl: cute.Shape,
        cluster_shape_mnk: cute.Shape,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Build the cluster layout and optional swizzle.

        ``problem_shape_ntile_mnl`` counts CTAs along M, N, and batch.
        ``cluster_shape_mnk`` counts CTAs in a cluster; its K extent must be 1.
        ``swizzle_size`` counts clusters; 1 disables swizzling. With swizzling,
        ``raster_along_m`` selects M-major (true) or N-major (false) order.
        Raise ValueError if the K extent is not 1 or the swizzle size is less than 1.
        """

        if cluster_shape_mnk[2] != 1:  # type: ignore[index]
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")  # type: ignore[index]
        if swizzle_size < 1:
            raise ValueError(f"expect swizzle_size >= 1, but get {swizzle_size}")

        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        # cluster_shape_mnk is kept for reconstruction
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]  # type: ignore[index]
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self._loc = loc

        # The default raster order is M-major (column-major).
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(
                self.problem_shape_ntile_mnl,
                cluster_shape_mnk[:2],  # type: ignore[index]
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )

        if swizzle_size > 1:
            problem_shape_ncluster_mnl = cute.round_up(
                self.problem_layout_ncluster_mnl.shape,
                (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),
            )

            if raster_along_m:
                self.problem_layout_ncluster_mnl = cute.make_layout(
                    (
                        problem_shape_ncluster_mnl[0],  # type: ignore[index]
                        (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),  # type: ignore[index, operator]
                        problem_shape_ncluster_mnl[2],  # type: ignore[index]
                    ),
                    stride=(
                        swizzle_size,
                        (1, swizzle_size * problem_shape_ncluster_mnl[0]),  # type: ignore[index]
                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],  # type: ignore[index, operator]
                    ),
                    loc=loc,
                    ip=ip,
                )
            else:
                self.problem_layout_ncluster_mnl = cute.make_layout(
                    (
                        (swizzle_size, problem_shape_ncluster_mnl[0] // swizzle_size),  # type: ignore[index, operator]
                        problem_shape_ncluster_mnl[1],  # type: ignore[index]
                        problem_shape_ncluster_mnl[2],  # type: ignore[index]
                    ),
                    stride=(
                        (1, swizzle_size * problem_shape_ncluster_mnl[1]),  # type: ignore[index]
                        swizzle_size,
                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],  # type: ignore[index, operator]
                    ),
                    loc=loc,
                    ip=ip,
                )

        # FastDivmod requires an unswizzled column-major layout.
        if swizzle_size == 1:
            _problem_layout_size = cute.size(self.problem_layout_ncluster_mnl, loc=loc, ip=ip)
            cluster_count_m = self.problem_layout_ncluster_mnl.shape[0]
            cluster_count_n = self.problem_layout_ncluster_mnl.shape[1]

            if raster_along_m:
                cluster_count_major = cluster_count_m
                cluster_count_minor = cluster_count_n
            else:
                cluster_count_major = cluster_count_n
                cluster_count_minor = cluster_count_m

            self.cluster_shape_major_fdd = cute.fast_divmod_create_divisor(
                cluster_count_major, loc=loc, ip=ip
            )

            self.cluster_shape_minor_fdd = cute.fast_divmod_create_divisor(
                cluster_count_minor, loc=loc, ip=ip
            )
        else:
            self.cluster_shape_major_fdd = None
            self.cluster_shape_minor_fdd = None

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values, self._values_pos = [], []
        for obj in [
            self.problem_shape_ntile_mnl,
            self._cluster_shape_mnk,
            self.swizzle_size,
            self.raster_along_m,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))

        # Carry the divisors to the device; None has no MLIR type.
        fastdivmod_values = []
        fastdivmod_indices = []

        for i, (fdd_name, fdd_obj) in enumerate(
            [
                ("cluster_shape_major_fdd", self.cluster_shape_major_fdd),
                ("cluster_shape_minor_fdd", self.cluster_shape_minor_fdd),
            ]
        ):
            if fdd_obj is not None:
                fdd_values = extract_mlir_values(fdd_obj)
                fastdivmod_values.extend(fdd_values)
                fastdivmod_indices.append(i)

        values += fastdivmod_values
        self._values_pos.append(
            len(fastdivmod_indices)
        )  # Store count of FastDivmod objects, not values
        self._fastdivmod_indices = fastdivmod_indices

        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "PersistentTileSchedulerParams":
        obj_list = []
        values_copy = list(values)

        for obj, n_items in zip(
            [
                self.problem_shape_ntile_mnl,
                self._cluster_shape_mnk,
                self.swizzle_size,
                self.raster_along_m,
            ],
            self._values_pos[:-1],  # Exclude FastDivmod count
        ):
            obj_list.append(new_from_mlir_values(obj, values_copy[:n_items]))
            values_copy = values_copy[n_items:]

        # Rebuild derived layouts in the device context.
        new_params = PersistentTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

        fdd_names = ["cluster_shape_major_fdd", "cluster_shape_minor_fdd"]

        if hasattr(self, "_fastdivmod_indices") and len(self._fastdivmod_indices) > 0:
            # Replace constructor-created divisors with the carried values.
            for j, original_index in enumerate(self._fastdivmod_indices):
                fdd_name = fdd_names[original_index]
                original_fdd = getattr(self, fdd_name)
                if original_fdd is not None and j < len(values_copy):
                    # Each FastDivmodDivisor has 1 MLIR value
                    reconstructed_fdd = new_from_mlir_values(original_fdd, [values_copy[j]])
                    setattr(new_params, fdd_name, reconstructed_fdd)

        return new_params


class StaticPersistentTileScheduler:
    """Assign tiles to persistent clusters in CUTLASS/CuTe kernels.

    ``params`` holds the cluster and problem layouts. ``num_persistent_clusters``
    is the launch capacity; ``cta_id_in_cluster`` locates this CTA within it.
    ``_current_work_linear_idx`` is the current cluster index, and
    ``_num_tiles_executed`` counts executed tiles.
    """

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))

        # The carried parameters include the FastDivmod divisors.
        values.extend(extract_mlir_values(self.params))

        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "StaticPersistentTileScheduler":
        assert len(values) >= 6
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])

        params_values = values[6:]
        new_params = new_from_mlir_values(self.params, params_values)

        return StaticPersistentTileScheduler(
            new_params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: tuple[Integer, Integer, Integer],
        grid_dim: tuple[Integer, Integer, Integer],
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> "StaticPersistentTileScheduler":
        """Create a scheduler from the CTA's 3-D block index and launch grid dimensions."""

        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # blockIdx.z indexes persistent clusters.
        current_work_linear_idx = Int32(bidz)

        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        num_tiles_executed = Int32(0)
        return StaticPersistentTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> WorkTileInfo:
        """Return the current tile coordinate and validity for this CTA in the cluster."""

        is_valid = current_work_linear_idx < cute.size(
            self.params.problem_layout_ncluster_mnl, loc=loc, ip=ip
        )

        if self.params.swizzle_size == 1:
            cur_cluster_coord = self._get_cluster_work_idx_with_fastdivmod(
                current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            # FastDivmod cannot decode a swizzled layout.
            cur_cluster_coord = self.params.problem_layout_ncluster_mnl.get_flat_coord(
                current_work_linear_idx, loc=loc, ip=ip
            )

        cur_tile_coord = tuple(
            cute.arch.make_warp_uniform(Int32(x) * Int32(z) + Int32(y))
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,  # type: ignore[arg-type]
                (*self.params.cluster_shape_mn, Int32(1)),
            )
        )

        return WorkTileInfo(cur_tile_coord, cute.arch.make_warp_uniform(is_valid))

    def _get_cluster_work_idx_with_fastdivmod(
        self,
        current_work_linear_idx: Int32,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> tuple[Int32, Int32, Int32]:
        """Decode a linear work index into cluster coordinates ``(m, n, l)``.

        Match ``problem_layout_ncluster_mnl.get_hier_coord()``: these are cluster
        coordinates, not tile coordinates. The layout must be unswizzled.
        """

        # The layout counts M clusters, N clusters, and batches.
        # Decode in order of increasing stride: major, minor, batch.
        cluster_minor_batch, cluster_major = divmod(
            current_work_linear_idx, self.params.cluster_shape_major_fdd
        )

        batch_l, cluster_minor = divmod(cluster_minor_batch, self.params.cluster_shape_minor_fdd)

        if self.params.raster_along_m:
            cluster_m = cluster_major
            cluster_n = cluster_minor
        else:
            cluster_m = cluster_minor
            cluster_n = cluster_major

        return (cluster_m, cluster_n, batch_l)

    @dsl_user_op
    @cute.jit
    def get_current_work(
        self,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(
        self,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(
        self,
        *,
        advance_count: int = 1,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._current_work_linear_idx += Int32(advance_count) * Int32(self.num_persistent_clusters)
        self._num_tiles_executed += Int32(1)
