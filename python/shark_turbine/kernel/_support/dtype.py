__all__ = [
    "DataType",
    "bool",
    "i4",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "f32",
    "f64",
    "index",
]

_INT_TYPES = ["i1", "i4", "i8", "i16", "i32", "i64"]
_FLOAT_TYPES = ["f16", "f32", "f64"]
_INDEX_TYPES = ["index"]


class DataType:
    name: str

    def __init__(self, name):
        self.name = name

    def ir_type_asm(self):
        return self.name

    def is_int(self):
        return self.name in _INT_TYPES

    def is_float(self):
        return self.name in _FLOAT_TYPES

    def is_index(self):
        return self.name in _INDEX_TYPES

    def is_bool(self):
        return self.name == "bool"


bool = DataType("bool")
i4 = DataType("i4")
i8 = DataType("i8")
i16 = DataType("i16")
i32 = DataType("i32")
i64 = DataType("i64")
f32 = DataType("f32")
f64 = DataType("f64")
f16 = DataType("f16")
f32 = DataType("f32")
f64 = DataType("f64")
index = DataType("index")
