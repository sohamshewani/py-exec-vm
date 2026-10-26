from __future__ annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

# -----------------------------------------------------------------------------
# Error Hierarchy & Diagnostics (System Invariant 5)
# -----------------------------------------------------------------------------

class VMError(Exception):
    """Base class for all py-exec-vm errors."""
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"[{line}:{column}] {message}")
        self.line = line
        self.column = column


class SyntaxError(VMError):
    """Raised when parser encounters malformed token streams."""
    pass


class CompilationError(VMError):
    """Raised during static analysis or bytecode verification failures."""
    pass


# -----------------------------------------------------------------------------
# Lexer Definitions
# -----------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Token:
    __slots__ = ("type", "value", "line", "column")
    type: str
    value: str
    line: int
    column: int


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.index = 0
        self.line = 1
        self.column = 1

    def _peek(self) -> str:
        if self.index >= self.length:
            return "\0"
        return self.source[self.index]

    def _advance(self) -> str:
        ch = self._peek()
        self.index += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _match(self, expected: str) -> bool:
        if self.index >= self.length:
            return False
        if self.source[self.index] != expected:
            return False
        self._advance()
        return True

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.index < self.length:
            ch = self._peek()
            if ch in " \t\r":
                self._advance()
                continue
            if ch == "\n":
                tokens.append(Token("NEWLINE", "\n", self.line, self.column))
                self._advance()
                continue
            if ch == "#":
                while self._peek() != "\n" and self._peek() != "\0":
                    self._advance()
                continue

            # Identifiers & Keywords
            if ch.isalpha() or ch == "_":
                start_line = self.line
                start_col = self.column
                val = []
                while self._peek().isalnum() or self._peek() == "_":
                    val.append(self._advance())
                text = "".join(val)
                kw = {
                    "let": "LET",
                    "fn": "FN",
                    "if": "IF",
                    "else": "ELSE",
                    "while": "WHILE",
                    "return": "RETURN",
                    "print": "PRINT",
                    "true": "TRUE",
                    "false": "FALSE",
                    "nil": "NIL",
                }
                t_type = kw.get(text, "IDENT")
                tokens.append(Token(t_type, text, start_line, start_col))
                continue

            # Numeric Literals
            if ch.isdigit():
                start_line = self.line
                start_col = self.column
                val = []
                while self._peek().isdigit():
                    val.append(self._advance())
                if self._peek() == "." and self.index + 1 < self.length and self.source[self.index + 1].isdigit():
                    val.append(self._advance())
                    while self._peek().isdigit():
                        val.append(self._advance())
                    tokens.append(Token("FLOAT", "".join(val), start_line, start_col))
                else:
                    tokens.append(Token("INT", "".join(val), start_line, start_col))
                continue

            # String Literals
            if ch == '"':
                start_line = self.line
                start_col = self.column
                self._advance()  # consume opening quote
                val = []
                while self._peek() != '"' and self._peek() != "\0":
                    val.append(self._advance())
                if self._peek() == "\0":
                    raise SyntaxError("Unterminated string literal", start_line, start_col)
                self._advance()  # consume closing quote
                tokens.append(Token("STRING", "".join(val), start_line, start_col))
                continue

            # Operators & Punctuation
            start_line = self.line
            start_col = self.column
            if ch == "=":
                self._advance()
                if self._match("="):
                    tokens.append(Token("EQ", "==", start_line, start_col))
                else:
                    tokens.append(Token("ASSIGN", "=", start_line, start_col))
            elif ch == "!";
                self._advance()
                if self._match("="):
                    tokens.append(Token("NEQ", "!=", start_line, start_col))
                else:
                    raise SyntaxError(f"Unexpected character '!'", start_line, start_col)
            elif ch == "<":
                self._advance()
                if self._match("="):
                    tokens.append(Token("LTE", "<=", start_line, start_col))
                else:
                    tokens.append(Token("LT", "<", start_line, start_col))
            elif ch == ">":
                self._advance()
                if self._match("="):
                    tokens.append(Token("GTE", ">=", start_line, start_col))
                else:
                    tokens.append(Token("GT", ">", start_line, start_col))
            elif ch == "+":
                self._advance()
                tokens.append(Token("PLUS", "+", start_line, start_col))
            elif ch == "-":
                self._advance()
                tokens.append(Token("MINUS", "-", start_line, start_col))
            elif ch == "*":
                self._advance()
                tokens.append(Token("STAR", "*", start_line, start_col))
            elif ch == "/":
                self._advance()
                tokens.append(Token("SLASH", "/", start_line, start_col))
            elif ch == "(":
                self._advance()
                tokens.append(Token("LPAREN", "(", start_line, start_col))
            elif ch == ")":
                self._advance()
                tokens.append(Token("RPAREN", ")", start_line, start_col))
            elif ch == "{":
                self._advance()
                tokens.append(Token("LBRACE", "{", start_line, start_col))
            elif ch == "}":
                self._advance()
                tokens.append(Token("RBRACE", "}", start_line, start_col))
            elif ch == ",":
                self._advance()
                tokens.append(Token("COMMA", ",", start_line, start_col))
            elif ch == ";":
                self._advance()
                tokens.append(Token("SEMICOLON", ";", start_line, start_col))
            else:
                raise SyntaxError(f"Unexpected character '{ch}'", start_line, start_col)

        tokens.append(Token("EOF", "", self.line, self.column))
        return tokens


# -----------------------------------------------------------------------------
# AST Definitions (System Invariant 1)
# -----------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ASTNode:
    __slots__ = ("line", "column")
    line: int
    column: int


@dataclass(slots=True, frozen=True)
class Program(ASTNode):
    __slots__ = ("body",)
    body: Tuple[ASTNode, ...]


@dataclass(slots=True, frozen=True)
class LetStmt(ASTNode):
    __slots__ = ("name", "initializer")
    name: str
    initializer: ASTNode


@dataclass(slots=True, frozen=True)
class AssignStmt(ASTNode):
    __slots__ = ("name", "value")
    name: str
    value: ASTNode


@dataclass(slots=True, frozen=True)
class FunctionDecl(ASTNode):
    __slots__ = ("name", "params", "body")
    name: str
    params: Tuple[str, ...]
    body: Tuple[ASTNode, ...]


@dataclass(slots=True, frozen=True)
class IfStmt(ASTNode):
    __slots__ = ("condition", "then_branch", "else_branch")
    condition: ASTNode
    then_branch: Tuple[ASTNode, ...]
    else_branch: Optional[Tuple[ASTNode, ...]]


@dataclass(slots=True, frozen=True)
class WhileStmt(ASTNode):
    __slots__ = ("condition", "body")
    condition: ASTNode
    body: Tuple[ASTNode, ...]


@dataclass(slots=True, frozen=True)
class ReturnStmt(ASTNode):
    __slots__ = ("value",)
    value: Optional[ASTNode]


@dataclass(slots=True, frozen=True)
class PrintStmt(ASTNode):
    __slots__ = ("expression",)
    expression: ASTNode


@dataclass(slots=True, frozen=True)
class ExprStmt(ASTNode):
    __slots__ = ("expression",)
    expression: ASTNode


@dataclass(slots=True, frozen=True)
class BinaryExpr(ASTNode):
    __slots__ = ("left", "operator", "right")
    left: ASTNode
    operator: str
    right: ASTNode


@dataclass(slots=True, frozen=True)
class CallExpr(ASTNode):
    __slots__ = ("callee", "arguments")
    callee: ASTNode
    arguments: Tuple[ASTNode, ...]


@dataclass(slots=True, frozen=True)
class VariableExpr(ASTNode):
    __slots__ = ("name",)
    name: str


@dataclass(slots=True, frozen=True)
class LiteralExpr(ASTNode):
    __slots__ = ("value", "type_tag")
    value: Union[int, float, str, bool, None]
    type_tag: str


# -----------------------------------------------------------------------------
# Parser Implementation (System Invariant 2 - Reentrant Execution Guards)
# -----------------------------------------------------------------------------

class Parser:
    MAX_RECURSION_DEPTH = 1000

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.current = 0
        self.recursion_depth = 0

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _is_at_end(self) -> bool:
        return self._peek().type == "EOF"

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self.tokens[self.current - 1]

    def _check(self, t_type: str) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == t_type

    def _match(self, *t_types: str) -> bool:
        for t_type in t_types:
            if self._check(t_type):
                self._advance()
                return True
        return False

    def _consume(self, t_type: str, message: str) -> Token:
        if self._check(t_type):
            return self._advance()
        tok = self._peek()
        raise SyntaxError(message, tok.line, tok.column)

    def _guard_recursion(self) -> None:
        self.recursion_depth += 1
        if self.recursion_depth > self.MAX_RECURSION_DEPTH:
            tok = self._peek()
            raise SyntaxError("Maximum parser recursion depth exceeded", tok.line, tok.column)

    def _ungaurd_recursion(self) -> None:
        self.recursion_depth -= 1

    def parse(self) -> Program:
        start_tok = self._peek()
        statements: List[ASTNode] = []
        while not self._is_at_end():
            self._match("NEWLINE")
            if self._is_at_end():
                break
            statements.append(self._declaration())
        return Program(body=tuple(statements), line=start_tok.line, column=start_tok.column)

    def _declaration(self) -> ASTNode:
        self._guard_recursion()
        try:
            if self._match("LET"):
                return self._let_declaration()
            if self._match("FN"):
                return self._function_declaration()
            return self._statement()
        finally:
            self._ungaurd_recursion()

    def _let_declaration(self) -> LetStmt:
        tok = self._peek()
        name_tok = self._consume("IDENT", "Expect variable name after 'let'")
        self._consume("ASSIGN", "Expect '=' after variable name in declaration")
        initializer = self._expression()
        self._match("SEMICOLON")
        self._match("NEWLINE")
        return LetStmt(name=name_tok.value, initializer=initializer, line=tok.line, column=tok.column)

    def _function_declaration(self) -> FunctionDecl:
        tok = self._peek()
        name_tok = self._consume("IDENT", "Expect function name after 'fn'")
        self._consume("LPAREN", "Expect '(' after function name")
        params: List[str] = []
        if not self._check("RPAREN"):
            while True:
                param_tok = self._consume("IDENT", "Expect parameter name")
                params.append(param_tok.value)
                if not self._match("COMMA"):
                    break
        self._consume("RPAREN", "Expect ')' after parameters")
        self._consume("LBRACE", "Expect '{' before function body")
        self._match("NEWLINE")
        body: List[ASTNode] = []
        while not self._check("RBRACE") and not self._is_at_end():
            self._match("NEWLINE")
            if self._check("RBRACE"):
                break
            body.append(self._declaration())
        self._consume("RBRACE", "Expect '}' after function body")
        self._match("NEWLINE")
        return FunctionDecl(name=name_tok.value, params=tuple(params), body=tuple(body), line=tok.line, column=tok.column)

    def _statement(self) -> ASTNode:
        if self._match("IF"):
            return self._if_statement()
        if self._match("WHILE"):
            return self._while_statement()
        if self._match("RETURN"):
            return self._return_statement()
        if self._match("PRINT"):
            return self._print_statement()
        
        # Check for assignment statement
        if self._check("IDENT") and self.current + 1 < len(self.tokens) and self.tokens[self.current + 1].type == "ASSIGN":
            name_tok = self._advance()
            self._advance()  # consume ASSIGN
            value = self._expression()
            self._match("SEMICOLON")
            self._match("NEWLINE")
            return AssignStmt(name=name_tok.value, value=value, line=name_tok.line, column=name_tok.column)

        return self._expression_statement()

    def _if_statement(self) -> IfStmt:
        tok = self._peek()
        self._consume("LPAREN", "Expect '(' after 'if'")
        condition = self._expression()
        self._consume("RPAREN", "Expect ')' after if condition")
        self._consume("LBRACE", "Expect '{' after if condition")
        self._match("NEWLINE")
        then_branch: List[ASTNode] = []
        while not self._check("RBRACE") and not self._is_at_end():
            self._match("NEWLINE")
            if self._check("RBRACE"):
                break
            then_branch.append(self._declaration())
        self._consume("RBRACE", "Expect '}' after if branch")
        self._match("NEWLINE")

        else_branch: Optional[Tuple[ASTNode, ...]] = None
        if self