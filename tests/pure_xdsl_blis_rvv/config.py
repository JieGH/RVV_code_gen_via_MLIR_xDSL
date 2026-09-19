import json
import os
from dataclasses import dataclass


@dataclass
class BLISProjectConfig:
    datatype: str  # e.g. "fp32", "int8", "i8i8i32"
    method: str  # e.g. "AXPY_FP32", "DOT_INT8"
    mr_nr_family: list[tuple[int, int]]  # e.g. [(8, 8), (32, 32)]
    vlen: int  # e.g. 256, 512
    arch: str  # e.g. "jbpi2"
    datasets: list[str] = None  # e.g. ["cache_test.dat", "special.dat"]
    skip_ref_run: bool = False
    iterations: int = 1000
    is_pinned: bool = False
    # Optional: {"label": forced_nc}. The dataset row whose label is exactly a key gets
    # its heuristic `nc` (from get_optim_mc_nc_kc) overridden with the given value,
    # clamped to the row's own N (issue #16: exact match, so B30 does not inherit B3).
    # Debug/mitigation knob for shape-specific instability (see issue #1) — does not
    # change the general cache-blocking model for any other shape. The reference design
    # has no such override, so a head-to-head run passes --no-nc-overrides to the driver.
    nc_overrides: dict[str, int] = None

    @classmethod
    def from_json(cls, file_path: str) -> list["BLISProjectConfig"]:
        import re

        with open(file_path) as f:
            lines = f.readlines()

        # Remove lines that are just comments
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Also handle inline comments (basic implementation)
            if "#" in line and '"' not in line.split("#")[1]:
                line = line.split("#")[0] + "\n"
            elif "//" in line and '"' not in line.split("//")[1]:
                line = line.split("//")[0] + "\n"
            cleaned_lines.append(line)

        json_str = "".join(cleaned_lines)

        # Remove trailing commas (which often happen when commenting out the last item)
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        data = json.loads(json_str)

        configs = []
        for row in data:
            is_pinned = (
                len(row) >= 8
            )  # As requested: 8th arg means PINNED, no 8th arg means SWEEP
            nc_overrides = row[8] if len(row) >= 9 else None

            if len(row) >= 8:
                (
                    datatype,
                    method,
                    mr_nr,
                    vlen,
                    arch,
                    datasets,
                    skip_ref_run,
                    iterations,
                ) = row[:8]
            elif len(row) == 7:
                datatype, method, mr_nr, vlen, arch, datasets, skip_ref_run = row
                iterations = 1000  # Fallback matches class default
            else:
                datatype, method, mr_nr, vlen, arch, datasets = row
                skip_ref_run = False
                iterations = 1000  # Fallback matches class default

            # Parse mr_nr into a list of tuples
            if len(mr_nr) == 2 and isinstance(mr_nr[0], int):
                mr_nr_family = [tuple(mr_nr)]
            else:
                mr_nr_family = [tuple(pair) for pair in mr_nr]

            configs.append(
                cls(
                    datatype=datatype,
                    method=method,
                    mr_nr_family=mr_nr_family,
                    vlen=vlen,
                    arch=arch,
                    datasets=datasets,
                    skip_ref_run=skip_ref_run,
                    iterations=iterations,
                    is_pinned=is_pinned,
                    nc_overrides=nc_overrides,
                )
            )

        return configs

    def to_json(self, file_path: str):
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def out_dir(self) -> str:
        """pj_gen/ output directory for this config.

        Pinned configs get an MR/NR suffix so that multiple pinned rows sharing
        the same (datatype, method, vlen) — e.g. issue #2's per-shape MR/NR
        matrix — land in distinct directories instead of silently overwriting
        each other in generate order.
        """
        out_dir = f"pj_gen/{self.datatype}_{self.method}_vlen{self.vlen}"
        if self.is_pinned:
            mr, nr = self.mr_nr_family[0]
            out_dir += f"_MR{mr}NR{nr}"
        return out_dir
