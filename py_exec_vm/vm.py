import sys
import time
import struct
import dataclasses
from typing import List, Dict, Tuple, Optional, Any, Set, Union


# ============================================================================
# 1. DATA STRUCTURES & REPRESENTATION
# ============================================================================

class VMError(Exception):
    """Base error class for py_exec_vm."""
    def __init__(self, message: str, code: int, line: int = 0, col: int = 0):
        super().__init__(message)
        self.message = message
        self.code = code
        self.line = line
        self.col = col

class CompileError(VMError): pass
class RuntimeErrorFault(VMError): pass
class TypeErrorFault(VMError): pass


@dataclasses.dataclass(frozen=True)
class ASTNode:
    __slots__ = ('line', 'col')
    line: int
    col: int

@dataclasses.dataclass(frozen=True)
class ASTLiteral(ASTNode):
    __slots__ = ('value',)
    value: Any

@dataclasses.dataclass(frozen=True)
class ASTIdentifier(ASTNode):
    __slots__ = ('name',)
    name: str

@dataclasses.dataclass(frozen=True)
class ASTBinaryOp(ASTNode):
    __slots__ = ('op', 'left', 'right')
    op: str
    left: ASTNode
    right: ASTNode

@dataclasses.dataclass(frozen=True)
class ASTAssign(ASTNode):
    __slots__ = ('name', 'expr')
    name: str
    expr: ASTNode

@dataclasses.dataclass(frozen=True)
class ASTIf(ASTNode):
    __slots__ = ('cond', 'then_branch', 'else_branch')
    cond: ASTNode
    then_branch: List[ASTNode]
    else_branch: List[ASTNode]

@dataclasses.dataclass(frozen=True)
class ASTWhile(ASTNode):
    __slots__ = ('cond', 'body')
    cond: ASTNode
    body: List[ASTNode]

@dataclasses.dataclass(frozen=True)
class ASTPrint(ASTNode):
    __slots__ = ('expr',)
    expr: ASTNode


# Bytecode Opcodes
OP_NOP     = 0x00
OP_LOADK   = 0x01
OP_MOVE    = 0x02
OP_ADD     = 0x03
OP_SUB     = 0x04
OP_MUL     = 0x05
OP_DIV     = 0x06
OP_EQ      = 0x07
OP_LT      = 0x08
OP_JMP     = 0x09
OP_JMPF    = 0x0A
OP_PRINT   = 0x0B
OP_HALT    = 0x0C
OP_LOADG   = 0x0D
OP_STOREG  = 0x0E


@dataclasses.dataclass(frozen=True)
class Instruction:
    __slots__ = ('opcode', 'reg_a', 'reg_b', 'reg_c', 'imm')
    opcode: int
    reg_a: int
    reg_b: int
    reg_c: int
    imm: int

    def encode(self) -> int:
        # Pack into 64-bit integer: Opcode(8), RegA(8), RegB(8), RegC(8), Imm(32, signed)
        imm_unsigned = self.imm & 0xFFFFFFFF
        return (
            (self.opcode & 0xFF) << 56 |
            (self.reg_a & 0xFF) << 48 |
            (self.reg_b & 0xFF) << 32 |
            (self.reg_c & 0xFF) << 24 |
            imm_unsigned
        )

    @staticmethod
    def decode(encoded: int) -> 'Instruction':
        opcode = (encoded >> 56) & 0xFF
        reg_a = (encoded >> 48) & 0xFF
        reg_b = (encoded >> 32) & 0xFF
        reg_c = (encoded >> 24) & 0xFF
        imm_unsigned = encoded & 0xFFFFFFFF
        if imm_unsigned & 0x80000000:
            imm = imm_unsigned - 0x100000000
        else:
            imm = imm_unsigned
        return Instruction(opcode, reg_a, reg_b, reg_c, imm)


class HeapObject:
    __slots__ = ('gc_marked', 'gc_generation', 'type_descriptor', 'forwarding_pointer', 'value', 'prev', 'next')
    def __init__(self, value: Any, type_desc: str):
        self.gc_marked: bool = False
        self.gc_generation: int = 0
        self.type_descriptor: str = type_desc
        self.forwarding_pointer: Optional['HeapObject'] = None
        self.value: Any = value
        self.prev: Optional['HeapObject'] = None
        self.next: Optional['HeapObject'] = None


class IntrusiveHeapList:
    __slots__ = ('head', 'tail', 'count')
    def __init__(self):
        self.head: Optional[HeapObject] = None
        self.tail: Optional[HeapObject] = None
        self.count: int = 0

    def push(self, obj: HeapObject) -> None:
        obj.prev = self.tail
        obj.next = None
        if self.tail is not None:
            self.tail.next = obj
        else:
            self.head = obj
        self.tail = obj
        self.count += 1

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
        self.count -= 1


class FlatSymbolTable:
    __slots__ = ('keys', 'values', 'capacity', 'size')
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.keys: List[Optional[str]] = [None] * capacity
        self.values: List[Any] = [None] * capacity
        self.size = 0

    def _hash(self, key: str) -> int:
        return hash(key) % self.capacity

    def put(self, key: str, value: Any) -> None:
        if self.size >= self.capacity * 0.75:
            self._resize()
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            idx = (idx + 1) % self.capacity
        self.keys[idx] = key
        self.values[idx] = value
        self.size += 1

    def get(self, key: str) -> Any:
        idx = self._hash(key)
        start_idx = idx
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.capacity
            if idx == start_idx:
                break
        raise RuntimeErrorFault(f"Symbol not found: {key}", 3001)

    def _resize(self) -> None:
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
        for i in range(len(old_keys)):
            if old_keys[i] is not None:
                self.put(old_keys[i], old_values[i])


# ============================================================================
# 2. CONCURRENCY PRIMITIVES & EXECUTION MODEL
# ============================================================================

class TaskQueue:
    __slots__ = ('queue',)
    def __init__(self):
        self.queue: List[Any] = []

    def push(self, item: Any) -> None:
        self.queue.append(item)

    def pop(self) -> Optional[Any]:
        if self.queue:
            return self.queue.pop(0)
        return None

    def __len__(self) -> int:
        return len(self.queue)


class CallFrameGuard:
    __slots__ = ('vm', 'max_depth')
    def __init__(self, vm: 'VirtualMachine', max_depth: int = 1024):
        self.vm = vm
        self.max_depth = max_depth

    def __enter__(self):
        if len(self.vm.call_stack) >= self.max_depth:
            raise RuntimeErrorFault("Call stack depth exceeded maximum bound", 2001)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ============================================================================
# 3. MEMORY LAYOUT & GARBAGE COLLECTION
# ============================================================================

class GarbageCollector:
    __slots__ = ('heap', 'allocation_quota', 'allocated_bytes')
    def __init__(self, allocation_quota: int = 1024 * 1024):
        self.heap = IntrusiveHeapList()
        self.allocation_quota = allocation_quota
        self.allocated_bytes = 0

    def allocate(self, value: Any, type_desc: str) -> HeapObject:
        self.allocated_bytes += 64  # nominal footprint per object
        if self.allocated_bytes >= self.allocation_quota:
            self.collect_garbage(set())
        obj = HeapObject(value, type_desc)
        self.heap.push(obj)
        return obj

    def collect_garbage(self, roots: Set[HeapObject]) -> None:
        worklist: List[HeapObject] = []
        # Phase 1: Mark roots (Grey phase initialization)
        for root in roots:
            if root and not root.gc_marked:
                root.gc_marked = True
                worklist.append(root)

        # Phase 2: Tri-color graph traversal using explicit worklist
        while worklist:
            obj = worklist.pop()
            # Inspect references inside obj.value if it contains HeapObjects
            if isinstance(obj.value, (list, tuple)):
                for item in obj.value:
                    if isinstance(item, HeapObject) and not item.gc_marked:
                        item.gc_marked = True
                        worklist.append(item)
            elif isinstance(obj.value, dict):
                for k, v in obj.value.items():
                    if isinstance(k, HeapObject) and not k.gc_marked:
                        k.gc_marked = True
                        worklist.append(k)
                    if isinstance(v, HeapObject) and not v.gc_marked:
                        v.gc_marked = True
                        worklist.append(v)

        # Phase 3: Sweep phase
        current = self.heap.head
        while current is not None:
            nxt = current.next
            if not current.gc_marked:
                self.heap.remove(current)
                self.allocated_bytes -= 64
            else:
                current.gc_marked = False
            current = nxt


# ============================================================================
# 4. SYSTEM INVARIANTS & SAFETY GUARANTEES
# ============================================================================

class Compiler:
    __slots__ = ('constants', 'instructions', 'globals_map')
    def __init__(self):
        self.constants: List[Any] = []
        self.instructions: List[int] = []
        self.globals_map: Dict[str, int] = {}

    def add_constant(self, val: Any) -> int:
        self.constants.append(val)
        return len(self.constants) - 1

    def compile(self, ast_nodes: List[ASTNode]) -> List[int]:
        for node in ast_nodes:
            self._compile_node(node)
        self.instructions.append(Instruction(OP_HALT, 0, 0, 0, 0).encode())
        return self.instructions

    def _compile_node(self, node: ASTNode) -> int:
        if isinstance(node, ASTLiteral):
            const_idx = self.add_constant(node.value)
            # Emit LOADK into Reg 0
            inst = Instruction(OP_LOADK, 0, 0, 0, const_idx)
            self.instructions.append(inst.encode())
            return 0
        elif isinstance(node, ASTIdentifier):
            if node.name not in self.globals_map:
                self.globals_map[node.name] = len(self.globals_map)
            g_idx = self.globals_map[node.name]
            inst = Instruction(OP_LOADG, 0, 0, 0, g_idx)
            self.instructions.append(inst.encode())
            return 0
        elif isinstance(node, ASTAssign):
            self._compile_node(node.expr)
            if node.name not in self.globals_map:
                self.globals_map[node.name] = len(self.globals_map)
            g_idx = self.globals_map[node.name]
            inst = Instruction(OP_STOREG, 0, 0, 0, g_idx)
            self.instructions.append(inst.encode())
            return 0
        elif isinstance(node, ASTBinaryOp):
            self._compile_node(node.left)
            # Move Reg 0 to Reg 1
            self.instructions.append(Instruction(OP_MOVE, 1, 0, 0, 0).encode())
            self._compile_node(node.right)
            # Move Reg 0 to Reg 2
            self.instructions.append(Instruction(OP_MOVE, 2, 0, 0, 0).encode())
            
            op_map = {
                '+': OP_ADD,
                '-': OP_SUB,
                '*': OP_MUL,
                '/': OP_DIV,
                '==': OP_EQ,
                '<': OP_LT
            }
            if node.op not in op_map:
                raise CompileError(f"Unsupported binary operator: {node.op}", 1001, node.line, node.col)
            
            inst = Instruction(op_map[node.op], 0, 1, 2, 0)
            self.instructions.append(inst.encode())
            return 0
        elif isinstance(node, ASTPrint):
            self._compile_node(node.expr)
            inst = Instruction(OP_PRINT, 0, 0, 0, 0)
            self.instructions.append(inst.encode())
            return 0
        elif isinstance(node, ASTIf):
            self._compile_node(node.cond)
            jmpf_idx = len(self.instructions)
            self.instructions.append(Instruction(OP_JMPF, 0, 0, 0, 0).encode())
            for n in node.then_branch:
                self._compile_node(n)
            jmp_idx = len(self.instructions)
            self.instructions.append(Instruction(OP_JMP, 0, 0, 0, 0).encode())
            
            else_target = len(self.instructions)
            # Patch JMPF offset
            self.instructions[jmpf_idx] = Instruction(OP_JMPF, 0, 0, 0, else_target).encode()
            
            for n in node.else_branch:
                self._compile_node(n)
            end_target = len(self.instructions)
            # Patch JMP offset
            self.instructions[jmp_idx] = Instruction(OP_JMP, 0, 0, 0, end_target).encode()
            return 0
        elif isinstance(node, ASTWhile):
            loop_start = len(self.instructions)
            self._compile_node(node.cond)
            jmpf_idx = len(self.instructions)
            self.instructions.append(Instruction(OP_JMPF, 0, 0, 0, 0).encode())
            for n in