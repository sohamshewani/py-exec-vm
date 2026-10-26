from __future__ annotations

import sys
import struct
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Final,
    Iterator,
)

# -----------------------------------------------------------------------------
# System Constants & Limits
# -----------------------------------------------------------------------------
MAX_REGISTERS: Final[int] = 256
MAX_STACK_DEPTH: Final[int] = 1024
INSTRUCTION_SIZE_BYTES: Final[int] = 8

# GC Color Constants
COLOR_WHITE: Final[int] = 0
COLOR_GREY: Final[int] = 1
COLOR_BLACK: Final[int] = 2

# -----------------------------------------------------------------------------
# Error Hierarchy
# -----------------------------------------------------------------------------
class VMError(Exception):
    """Base exception class for all py-exec-vm errors."""
    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

class SyntaxError(VMError):
    """Raised when parser encounters malformed syntax."""
    pass

class CompilationError(VMError):
    """Raised when the bytecode compiler detects invalid AST or type invariants."""
    pass

class RuntimeError(VMError):
    """Raised during VM execution on unrecoverable instruction faults."""
    pass

class MemoryError(VMError):
    """Raised when allocation thresholds are breached or out-of-memory occurs."""
    pass


# -----------------------------------------------------------------------------
# 1. AST Nodes (Immutable with __slots__)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceLocation:
    __slots__ = ("line", "column")
    line: int
    column: int

@dataclass(frozen=True)
class ASTNode:
    __slots__ = ("location",)
    location: SourceLocation

@dataclass(frozen=True)
class LiteralNode(ASTNode):
    __slots__ = ("value",)
    value: Any

@dataclass(frozen=True)
class IdentifierNode(ASTNode):
    __slots__ = ("name",)
    name: str

@dataclass(frozen=True)
class BinaryOpNode(ASTNode):
    __slots__ = ("operator", "left", "right")
    operator: str
    left: ASTNode
    right: ASTNode

@dataclass(frozen=True)
class AssignmentNode(ASTNode):
    __slots__ = ("target", "value")
    target: IdentifierNode
    value: ASTNode

@dataclass(frozen=True)
class IfNode(ASTNode):
    __slots__ = ("condition", "consequence", "alternative")
    condition: ASTNode
    consequence: Sequence[ASTNode]
    alternative: Optional[Sequence[ASTNode]]

@dataclass(frozen=True)
class WhileNode(ASTNode):
    __slots__ = ("condition", "body")
    condition: ASTNode
    body: Sequence[ASTNode]

@dataclass(frozen=True)
class FunctionDefNode(ASTNode):
    __slots__ = ("name", "parameters", "body")
    name: str
    parameters: Sequence[str]
    body: Sequence[ASTNode]

@dataclass(frozen=True)
class ReturnNode(ASTNode):
    __slots__ = ("value",)
    value: Optional[ASTNode]


# -----------------------------------------------------------------------------
# 2. Bytecode Instruction Set (Fixed-width 64-bit encoding)
# -----------------------------------------------------------------------------
class Opcode:
    LOAD_CONST: Final[int] = 0x01
    LOAD_VAR: Final[int] = 0x02
    STORE_VAR: Final[int] = 0x03
    ADD: Final[int] = 0x04
    SUB: Final[int] = 0x05
    MUL: Final[int] = 0x06
    DIV: Final[int] = 0x07
    JUMP: Final[int] = 0x08
    JUMP_IF_FALSE: Final[int] = 0x09
    CALL: Final[int] = 0x0A
    RETURN: Final[int] = 0x0B
    HALT: Final[int] = 0x0C

@dataclass(frozen=True)
class Instruction:
    """
    Fixed-width 64-bit encoded instruction layout:
    [ Opcode: 8 bits ][ RegA: 8 bits ][ RegB: 8 bits ][ RegC: 8 bits ][ Immediate: 32 bits ]
    Packed format string: '!B B B B I' (Big-endian, 8 bytes total)
    """
    __slots__ = ("opcode", "reg_a", "reg_b", "reg_c", "immediate")

    opcode: int
    reg_a: int
    reg_b: int
    reg_c: int
    immediate: int

    def encode(self) -> bytes:
        if not (0 <= self.opcode <= 0xFF):
            raise CompilationError(f"Opcode out of bounds: {self.opcode}")
        if not (0 <= self.reg_a <= 0xFF):
            raise CompilationError(f"RegA out of bounds: {self.reg_a}")
        if not (0 <= self.reg_b <= 0xFF):
            raise CompilationError(f"RegB out of bounds: {self.reg_b}")
        if not (0 <= self.reg_c <= 0xFF):
            raise CompilationError(f"RegC out of bounds: {self.reg_c}")
        if not (0 <= self.immediate <= 0xFFFFFFFF):
            raise CompilationError(f"Immediate out of bounds: {self.immediate}")
        
        return struct.pack("!BBBBW", self.opcode, self.reg_a, self.reg_b, self.reg_c, self.immediate)

    @classmethod
    def decode(cls, raw_bytes: bytes) -> Instruction:
        if len(raw_bytes) != INSTRUCTION_SIZE_BYTES:
            raise RuntimeError(f"Invalid instruction byte length: {len(raw_bytes)}")
        opcode, reg_a, reg_b, reg_c, immediate = struct.unpack("!BBBBW", raw_bytes)
        return cls(opcode, reg_a, reg_b, reg_c, immediate)


# -----------------------------------------------------------------------------
# 3. Heap Objects & Memory Management
# -----------------------------------------------------------------------------
class HeapObject:
    """
    Heap object header tracking GC reachability, type descriptors, and 
    intrusive doubly-linked list pointers.
    """
    __slots__ = (
        "gc_color",
        "type_descriptor",
        "forwarding_pointer",
        "value",
        "prev",
        "next",
    )

    def __init__(self, type_descriptor: str, value: Any) -> None:
        self.gc_color: int = COLOR_WHITE
        self.type_descriptor: str = type_descriptor
        self.forwarding_pointer: Optional[HeapObject] = None
        self.value: Any = value
        self.prev: Optional[HeapObject] = None
        self.next: Optional[HeapObject] = None

class UnmanagedHeapArena:
    """
    Arena-based chunk allocation strategy grouping objects by size class 
    with intrusive doubly-linked list tracking.
    """
    __slots__ = ("head", "tail", "allocation_count", "capacity_limit", "allocated_bytes")

    def __init__(self, capacity_limit: int = 1048576) -> None:
        self.head: Optional[HeapObject] = None
        self.tail: Optional[HeapObject] = None
        self.allocation_count: int = 0
        self.capacity_limit: int = capacity_limit
        self.allocated_bytes: int = 0

    def allocate(self, type_descriptor: str, value: Any) -> HeapObject:
        if self.allocation_count >= self.capacity_limit:
            raise MemoryError("Heap arena allocation threshold breached. GC safepoint required.")
        
        obj = HeapObject(type_descriptor, value)
        
        if self.head is None:
            self.head = obj
            self.tail = obj
        else:
            if self.tail is not None:
                self.tail.next = obj
                obj.prev = self.tail
                self.tail = obj

        self.allocation_count += 1
        self.allocated_bytes += sys.getsizeof(value)
        return obj

    def remove(self, obj: HeapObject) -> None:
        if obj.prev is not None:
            obj.prev.next = obj.next
        else:
            self.head = obj.next

        if obj.next is not None:
            obj.next.prev = obj.prev
        else:
            self.tail = obj.prev

        obj.prev = None
        obj.next = None
        self.allocation_count = max(0, self.allocation_count - 1)
        self.allocated_bytes = max(0, self.allocated_bytes - sys.getsizeof(obj.value))

    def iter_objects(self) -> Iterator[HeapObject]:
        current = self.head
        while current is not None:
            yield current
            current = current.next


# -----------------------------------------------------------------------------
# 4. Symbol Tables & Environments (Flat-array linear probing)
# -----------------------------------------------------------------------------
class LexicalEnvironment:
    """
    Scoped lexical hash map implemented with flat-array linear probing 
    to eliminate deep call stack overhead.
    """
    __slots__ = ("keys", "values", "capacity", "size", "parent")

    def __init__(self, capacity: int = 64, parent: Optional[LexicalEnvironment] = None) -> None:
        self.capacity = capacity
        self.size = 0
        self.keys: List[Optional[str]] = [None] * capacity
        self.values: List[Any] = [None] * capacity
        self.parent = parent

    def _hash(self, key: str) -> int:
        return hash(key) % self.capacity

    def put(self, key: str, value: Any) -> None:
        if self.size >= self.capacity * 0.75:
            self._resize()

        index = self._hash(key)
        start_index = index

        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return
            index = (index + 1) % self.capacity
            if index == start_index:
                raise MemoryError("Lexical environment flat-array capacity exhausted.")

        self.keys[index] = key
        self.values[index] = value
        self.size += 1

    def get(self, key: str) -> Any:
        index = self._hash(key)
        start_index = index

        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.capacity
            if index == start_index:
                break

        if self.parent is not None:
            return self.parent.get(key)

        raise RuntimeError(f"Undefined symbol reference: '{key}'")

    def _resize(self) -> None:
        old_capacity = self.capacity
        old_keys = self.keys
        old_values = self.values

        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity

        for i in range(old_capacity):
            k = old_keys[i]
            if k is not None:
                self.put(k, old_values[i])


# -----------------------------------------------------------------------------
# 5. Execution Call Frame & VM Runtime Context
# -----------------------------------------------------------------------------
@dataclass
class CallFrame:
    __slots__ = ("function_name", "return_address", "environment", "registers")
    function_name: str
    return_address: int
    environment: LexicalEnvironment
    registers: List[Any]


# -----------------------------------------------------------------------------
# 6. Telemetry & Performance Counters
# -----------------------------------------------------------------------------
class TelemetryCollector:
    __slots__ = (
        "instructions_executed",
        "gc_pause_total_ms",
        "gc_runs",
        "heap_allocations",
        "cache_hits",
        "cache_misses",
    )

    def __init__(self) -> None:
        self.instructions_executed: int = 0
        self.gc_pause_total_ms: float = 0.0
        self.gc_runs: int = 0
        self.heap_allocations: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def record_instruction(self) -> None:
        self.instructions_executed += 1

    def record_gc(self, duration_ms: float) -> None:
        self.gc_runs += 1
        self.gc_pause_total_ms += duration_ms

    def snapshot(self) -> Dict[str, Any]:
        return {
            "instructions_executed": self.instructions_executed,
            "gc_pause_total_ms": self.gc_pause_total_ms,
            "gc_runs": self.gc_runs,
            "heap_allocations": self.heap_allocations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------
__all__ = [
    "MAX_REGISTERS",
    "MAX_STACK_DEPTH",
    "INSTRUCTION_SIZE_BYTES",
    "COLOR_WHITE",
    "COLOR_GREY",
    "COLOR_BLACK",
    "VMError",
    "SyntaxError",
    "CompilationError",
    "RuntimeError",
    "MemoryError",
    "SourceLocation",
    "ASTNode",
    "LiteralNode",
    "IdentifierNode",
    "BinaryOpNode",
    "AssignmentNode",
    "IfNode",
    "WhileNode",
    "FunctionDefNode",
    "ReturnNode",
    "Opcode",
    "Instruction",
    "HeapObject",
    "UnmanagedHeapArena",
    "LexicalEnvironment",
    "CallFrame",
    "TelemetryCollector",
]