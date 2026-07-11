import json
import os
from dataclasses import dataclass


@dataclass
class BLISProjectConfig:
    datatype: str  # "fp32" | "int8"
    method: str  # "AXPY_FP32_Pack" | "DOT_INT8_LANE" | ...
    vlen: int  # 256 | 512
    arch: str  # "jbpi2" | "generic_rvv" (points to cache configs)
    mr_nr_family: list[tuple[int, int]]  # e.g. [(4,4), (4,8), (8,8), (8,16)]
    datasets: list[str] = None  # e.g. ["cache_test.dat", "special.dat"]

    @classmethod
    def from_json(cls, file_path: str) -> "BLISProjectConfig":
        with open(file_path) as f:
            data = json.load(f)

        # Convert mr_nr_family list of lists to list of tuples
        if "mr_nr_family" in data:
            data["mr_nr_family"] = [tuple(pair) for pair in data["mr_nr_family"]]

        return cls(**data)

    def to_json(self, file_path: str):
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
