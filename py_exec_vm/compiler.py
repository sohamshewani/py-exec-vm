import sys
import struct
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# ==========================================
# CONSTANTS & OPCODES
# ==========================================

OP_NOP      = 0x00
OP_LOAD_CONST = 0x01
OP_LOAD_NAME  = 0x02
OP_STORE_NAME = 0x03
OP_ADD      = 0x04
OP_SUB      = 0x05
OP_MUL      = 0x06
OP_DIV      = 0x07
OP_JUMP     = 0x08
OP_JUMP_IF_FALSE = 0x09
OP_CALL     = 0x0A
OP_RETURN   = 0x0B
OP_PRINT    = 0x0C
OP_HALT     = 0x0D

# ==========================================
# AST NODES (Immutable with __slots__)
# ==========================================

class ASTNode:
    __slots__ = ()

@dataclass(frozen=True)
class LiteralNode(ASTNode):
    __slots__ = ('value',)
    value: Any

@dataclass(frozen=True)
class IdentifierNode(ASTNode):
    __slots__ = ('name',)
    name: str

@dataclass(frozen=True)
class BinaryOpNode(ASTNode):
    __slots__ = ('op', 'left', 'right')
    op: str
    left: ASTNode
    right: ASTNode

@dataclass(frozen=True)
class AssignNode(ASTNode):
    __slots__ = ('name', 'value')
    name: str
    value: ASTNode

@dataclass(frozen=True)
class PrintNode(ASTNode):
    __slots__ = ('expr',)
    expr: ASTNode

@dataclass(frozen=True)
class IfNode(ASTNode):
    __slots__ = ('condition', 'body', 'orelse')
    body: Tuple[ASTNode, ...]
    orelse: Tuple[ASTNode, ...]
    condition: ASTNode

@dataclass(frozen=True)
class WhileNode(ASTNode):
    __slots__ = ('condition', 'body')
    condition: ASTNode
    body: Tuple[ASTNode, ...]

@dataclass(frozen=True)
class BlockNode(ASTNode):
    __slots__ = ('statements',)
    statements: Tuple[ASTNode, ...]

# ==========================================
# BYTECODE ENCODER & CONTAINER
# ==========================================

class InstructionEncoder:
    @staticmethod
    def encode(opcode: int, reg_a: int, reg_b: int, reg_c: int, imm: int) -> int:
        """
        Fixed-width 64-bit instruction encoding:
        [ Opcode: 8 bits ][ RegA: 8 bits ][ RegB: 8 bits ][ RegC: 8 bits ][ Immediate: 32 bits ]
        """
        if not (0 <= opcode <= 0xFF):
            raise ValueError(f"Opcode out of bounds: {opcode}")
        if not (0 <= reg_a <= 0xFF):
            raise ValueError(f"RegA out of bounds: {reg_a}")
        if not (0 <= reg_b <= 0xFF):
            raise ValueError(f"RegB out of bounds: {reg_b}")
        if not (0 <= reg_c <= 0xFF):
            raise ValueError(f"RegC out of bounds: {reg_c}")
        if not (-0x80000000 <= imm <= 0x7FFFFFFF):
            imm = imm & 0xFFFFFFFF

        return (
            (opcode << 56) |
            (reg_a << 48) |
            (reg_b << 40) |
            (reg_c << 32) |
            imm
        )

@dataclass
class CodeObject:
    __slots__ = ('code', 'constants', 'names', 'line_table')
    code: List[int]
    constants: List[Any]
    names: List[str]
    line_table: List[Tuple[int, int]]

# ==========================================
# COMPILER IMPLEMENTATION
# ==========================================

class CompilerError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(f"CompilerError: {message} at line {line}, col {col}")
        self.line = line
        self.col = col

class Compiler:
    def __init__(self) -> None:
        self.instructions: List[int] = []
        self.constants: List[Any] = []
        self.names: List[str] = []
        self.line_table: List[Tuple[int, int]] = []
        self._next_register: int = 0
        self._symbol_table: Dict[str, int] = {}

    def _alloc_register(self) -> int:
        reg = self._next_register
        if reg > 255:
            raise CompilerError("Register allocation overflow: exceeded 256 physical registers.")
        self._next_register += 1
        return reg

    def _free_register(self) -> None:
        if self._next_register > 0:
            self._next_register -= 1

    def _emit(self, opcode: int, reg_a: int, reg_b: int, reg_c: int, imm: int, line: int = 0) -> int:
        encoded = InstructionEncoder.encode(opcode, reg_a, reg_b, reg_c, imm)
        self.instructions.append(encoded)
        pc = len(self.instructions) - 1
        self.line_table.append((pc, line))
        return pc

    def _add_constant(self, value: Any) -> int:
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1

    def _add_name(self, name: str) -> int:
        if name in self.names:
            return self.names.index(name)
        self.names.append(name)
        return len(self.names) - 1

    def compile(self, node: ASTNode) -> CodeObject:
        self._visit(node)
        self._emit(OP_HALT, 0, 0, 0, 0)
        return CodeObject(
            code=self.instructions,
            constants=self.constants,
            names=self.names,
            line_table=self.line_table
        )

    def _visit(self, node: ASTNode) -> int:
        method_name = f"_visit_{type(node).__name__}"
        visitor = getattr(self, method_name, None)
        if visitor is None:
            raise CompilerError(f"No compilation visitor defined for {type(node).__name__}")
        return visitor(node)

    def _visit_LiteralNode(self, node: LiteralNode) -> int:
        const_idx = self._add_constant(node.value)
        dest_reg = self._alloc_register()
        self._emit(OP_LOAD_CONST, dest_reg, 0, 0, const_idx)
        return dest_reg

    def _visit_IdentifierNode(self, node: IdentifierNode) -> int:
        name_idx = self._add_name(node.name)
        dest_reg = self._alloc_register()
        self._emit(OP_LOAD_NAME, dest_reg, 0, 0, name_idx)
        return dest_reg

    def _visit_BinaryOpNode(self, node: BinaryOpNode) -> int:
        left_reg = self._visit(node.left)
        right_reg = self._visit(node.right)
        dest_reg = self._alloc_register()

        op_map = {
            '+': OP_ADD,
            '-': OP_SUB,
            '*': OP_MUL,
            '/': OP_DIV
        }
        
        if node.op not in op_map:
            raise CompilerError(f"Unsupported binary operator: {node.op}")

        self._emit(op_map[node.op], dest_reg, left_reg, right_reg, 0)
        
        self._free_register()
        self._free_register()
        return dest_reg

    def _visit_AssignNode(self, node: AssignNode) -> int:
        val_reg = self._visit(node.value)
        name_idx = self._add_name(node.name)
        self._emit(OP_STORE_NAME, val_reg, 0, 0, name_idx)
        return val_reg

    def _visit_PrintNode(self, node: PrintNode) -> int:
        expr_reg = self._visit(node.expr)
        self._emit(OP_PRINT, expr_reg, 0, 0, 0)
        self._free_register()
        return 0

    def _visit_IfNode(self, node: IfNode) -> int:
        cond_reg = self._visit(node.condition)
        
        jump_if_false_pc = self._emit(OP_JUMP_IF_FALSE, cond_reg, 0, 0, 0)
        self._free_register()

        for stmt in node.body:
            self._visit(stmt)

        jump_over_else_pc = self._emit(OP_JUMP, 0, 0, 0, 0)

        else_target_pc = len(self.instructions)
        
        # Patch JUMP_IF_FALSE
        orig_instr = self.instructions[jump_if_false_pc]
        opcode = (orig_instr >> 56) & 0xFF
        reg_a = (orig_instr >> 48) & 0xFF
        reg_b = (orig_instr >> 40) & 0xFF
        reg_c = (orig_instr >> 32) & 0xFF
        self.instructions[jump_if_false_pc] = InstructionEncoder.encode(opcode, reg_a, reg_b, reg_c, else_target_pc)

        for stmt in node.orelse:
            self._visit(stmt)

        end_target_pc = len(self.instructions)

        # Patch JUMP over else
        orig_instr = self.instructions[jump_over_else_pc]
        opcode = (orig_instr >> 56) & 0xFF
        reg_a = (orig_instr >> 48) & 0xFF
        reg_b = (orig_instr >> 40) & 0xFF
        reg_c = (orig_instr >> 32) & 0xFF
        self.instructions[jump_over_else_pc] = InstructionEncoder.encode(opcode, reg_a, reg_b, reg_c, end_target_pc)

        return 0

    def _visit_WhileNode(self, node: WhileNode) -> int:
        loop_start_pc = len(self.instructions)

        cond_reg = self._visit(node.condition)
        jump_exit_pc = self._emit(OP_JUMP_IF_FALSE, cond_reg, 0, 0, 0)
        self._free_register()

        for stmt in node.body:
            self._visit(stmt)

        self._emit(OP_JUMP, 0, 0, 0, loop_start_pc)

        loop_end_pc = len(self.instructions)

        orig_instr = self.instructions[jump_exit_pc]
        opcode = (orig_instr >> 56) & 0xFF
        reg_a = (orig_instr >> 48) & 0xFF
        reg_b = (orig_instr >> 40) & 0xFF
        reg_c = (orig_instr >> 32) & 0xFF
        self.instructions[jump_exit_pc] = InstructionEncoder.encode(opcode, reg_a, reg_b, reg_c, loop_end_pc)

        return 0

    def _visit_BlockNode(self, node: BlockNode) -> int:
        for stmt in node.statements:
            self._visit(stmt)
        return 0

# ==========================================
# MODULE SELF-TEST & VERIFICATION
# ==========================================

if __name__ == "__main__":
    # Test suite compilation verification
    ast_root = BlockNode(statements=(
        AssignNode(name="x", value=LiteralNode(value=10)),
        AssignNode(name="y", value=BinaryOpNode(op='+', left=IdentifierNode(name="x"), right=LiteralNode(value=5))),
        PrintNode(expr=IdentifierNode(name="y")),
        IfNode(
            condition=BinaryOpNode(op='-', left=IdentifierNode(name="y"), right=LiteralNode(value=10)),
            body=(PrintNode(expr=LiteralNode(value=999)),),
            orelse=(PrintNode(expr=LiteralNode(value=111)),)
        ),
        WhileNode(
            condition=BinaryOpNode(op='-', left=IdentifierNode(name="x"), right=LiteralNode(value=0)),
            body=(
                AssignNode(name="x", value=BinaryOpNode(op='-', left=IdentifierNode(name="x"), right=LiteralNode(value=1)))
            )
        )
    ))

    compiler = Compiler()
    code_obj = compiler.compile(ast_root)
    
    assert len(code_obj.code) > 0, "Compilation generated empty bytecode."
    assert len(code_obj.constants) > 0, "Compilation failed to collect constants."
    assert len(code_obj.names) > 0, "Compilation failed to collect names."
    
    print(f"Compilation successful. Generated {len(code_obj.code)} instructions, {len(code_obj.constants)} constants, {len(code_obj.names)} symbols.")