# py_exec_vm/lexer.py
"""
Production Lexer implementation for 'py-exec-vm'.
Transforms source text streams into an immutable, tightly packed stream of Tokens
complete with precise line/column tracking and structured error diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Final, Iterator, List, Optional, Tuple


class LexicalError(Exception):
    """Structured lexical error with exact line and column tracking."""

    __slots__ = ("message", "line", "column", "filename")

    def __init__(self, message: str, line: int, column: int, filename: str = "<string>") -> None:
        super().__init__(f"{filename}:{line}:{column}: error: {message}")
        self.message: Final[str] = message
        self.line: Final[int] = line
        self.column: Final[int] = column
        self.filename: Final[str] = filename


class TokenType(Enum):
    # Keywords
    DEF = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    CLASS = auto()
    PASS_KW = auto()
    BREAK = auto()
    CONTINUE = auto()
    GLOBAL = auto()
    NONLOCAL = auto()
    TRY = auto()
    EXCEPT = auto()
    FINALLY = auto()
    RAISE = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    LAMBDA = auto()
    ASSERT = auto()
    DEL = auto()
    YIELD = auto()
    ASYNC = auto()
    AWAIT = auto()

    # Literals
    IDENTIFIER = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()

    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    DOUBLE_SLASH = auto()  # //
    PERCENT = auto()       # %
    DOUBLE_STAR = auto()   # **
    EQ = auto()            # =
    EQ_EQ = auto()         # ==
    NOT_EQ = auto()        # !=
    LESS = auto()          # <
    LESS_EQ = auto()       # <=
    GREATER = auto()       # >
    GREATER_EQ = auto()    # >=
    PLUS_EQ = auto()       # +=
    MINUS_EQ = auto()      # -=
    STAR_EQ = auto()       # *=
    SLASH_EQ = auto()      # /=
    ARROW = auto()         # ->

    # Delimiters
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    COMMA = auto()         # ,
    COLON = auto()         # :
    DOT = auto()           # .
    SEMICOLON = auto()     # ;
    AT = auto()            # @

    # Structural
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass(frozen=True, slots=True)
__all__ = ["Token", "Lexer", "LexicalError", "TokenType"]


@dataclass(frozen=True, slots=True)
class Token:
    """Immutable, cache-friendly token representation using slots."""
    type: TokenType
    lexeme: str
    line: int
    column: int
    filename: str

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.lexeme!r}, line={self.line}, col={self.column})"


class Lexer:
    """Production-grade lexical analyzer with indentation tracking and comprehensive error handling."""

    __slots__ = (
        "_source",
        "_filename",
        "_length",
        "_index",
        "_line",
        "_column",
        "_indent_stack",
        "_tokens",
        "_at_bol",
        "_paren_depth",
    )

    KEYWORDS: Final[Dict[str, TokenType]] = {
        "def": TokenType.DEF,
        "return": TokenType.RETURN,
        "if": TokenType.IF,
        "else": TokenType.ELSE,
        "while": TokenType.WHILE,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "and": TokenType.AND,
        "or": TokenType.OR,
        "not": TokenType.NOT,
        "True": TokenType.TRUE,
        "False": TokenType.FALSE,
        "None": TokenType.NONE,
        "class": TokenType.CLASS,
        "pass": TokenType.PASS_KW,
        "break": TokenType.BREAK,
        "continue": TokenType.CONTINUE,
        "global": TokenType.GLOBAL,
        "nonlocal": TokenType.NONLOCAL,
        "try": TokenType.TRY,
        "except": TokenType.EXCEPT,
        "finally": TokenType.FINALLY,
        "raise": TokenType.RAISE,
        "import": TokenType.IMPORT,
        "from": TokenType.FROM,
        "as": TokenType.AS,
        "lambda": TokenType.LAMBDA,
        "assert": TokenType.ASSERT,
        "del": TokenType.DEL,
        "yield": TokenType.YIELD,
        "async": TokenType.ASYNC,
        "await": TokenType.AWAIT,
    }

    def __init__(self, source: str, filename: str = "<string>") -> None:
        self._source: Final[str] = source
        self._filename: Final[str] = filename
        self._length: Final[int] = len(source)
        self._index: int = 0
        self._line: int = 1
        self._column: int = 1
        self._indent_stack: List[int] = [0]
        self._tokens: List[Token] = []
        self._at_bol: bool = True
        self._paren_depth: int = 0

    def tokenize(self) -> List[Token]:
        """Executes the complete lexing pass, generating all tokens including INDENT/DEDENT."""
        while not self._is_at_end():
            self._scan_token()

        # Flush remaining indents at EOF
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self._tokens.append(Token(TokenType.DEDENT, "", self._line, self._column, self._filename))

        self._tokens.append(Token(TokenType.EOF, "", self._line, self._column, self._filename))
        return self._tokens

    def _is_at_end(self) -> bool:
        return self._index >= self._length

    def _peek(self) -> str:
        if self._is_at_end():
            return "\0"
        return self._source[self._index]

    def _peek_next(self) -> str:
        if self._index + 1 >= self._length:
            return "\0"
        return self._source[self._index + 1]

    def _advance(self) -> str:
        if self._is_at_end():
            return "\0"
        char = self._source[self._index]
        self._index += 1
        if char == "\n":
            self._line += 1
            self._column = 1
            self._at_bol = True
        else:
            self._column += 1
        return char

    def _match(self, expected: str) -> bool:
        if self._is_at_end() or self._source[self._index] != expected:
            return False
        self._advance()
        return True

    def _add_token(self, token_type: TokenType, lexeme: str, start_line: int, start_col: int) -> None:
        self._tokens.append(Token(token_type, lexeme, start_line, start_col, self._filename))

    def _scan_token(self) -> None:
        start_line = self._line
        start_col = self._column

        # Handle indentation at the beginning of a line
        if self._at_bol and self._paren_depth == 0:
            self._handle_indentation()
            if self._is_at_end():
                return
            start_line = self._line
            start_col = self._column

        c = self._advance()

        # Skip horizontal whitespace and explicit line continuation
        if c == " " or c == "\t" or c == "\r":
            return
        if c == "\\":
            if self._peek() == "\n":
                self._advance()
                return
            raise LexicalError("Unexpected character after line continuation '\\'", start_line, start_col, self._filename)

        if c == "\n":
            if self._paren_depth == 0:
                self._add_token(TokenType.NEWLINE, "\n", start_line, start_col)
            self._at_bol = True
            return

        # Comments
        if c == "#":
            while self._peek() != "\n" and not self._is_at_end():
                self._advance()
            return

        # Identifiers & Keywords
        if self._is_alpha(c):
            self._scan_identifier_or_keyword(start_line, start_col, c)
            return

        # Numeric Literals
        if self._is_digit(c):
            self._scan_number(start_line, start_col, c)
            return

        # String Literals
        if c in ('"', "'"):
            self._scan_string(start_line, start_col, c)
            return

        # Operators & Punctuation
        if c == "+":
            tt = TokenType.PLUS_EQ if self._match("=") else TokenType.PLUS
            self._add_token(tt, "+=" if tt == TokenType.PLUS_EQ else "+", start_line, start_col)
        elif c == "-":
            if self._match("="):
                self._add_token(TokenType.MINUS_EQ, "-=", start_line, start_col)
            elif self._match(">"):
                self._add_token(TokenType.ARROW, "->", start_line, start_col)
            else:
                self._add_token(TokenType.MINUS, "-", start_line, start_col)
        elif c == "*":
            if self._match("*"):
                self._add_token(TokenType.DOUBLE_STAR, "**", start_line, start_col)
            elif self._match("="):
                self._add_token(TokenType.STAR_EQ, "*=", start_line, start_col)
            else:
                self._add_token(TokenType.STAR, "*", start_line, start_col)
        elif c == "/":
            if self._match("/"):
                self._add_token(TokenType.DOUBLE_SLASH, "//", start_line, start_col)
            elif self._match("="):
                self._add_token(TokenType.SLASH_EQ, "/=", start_line, start_col)
            else:
                self._add_token(TokenType.SLASH, "/", start_line, start_col)
        elif c == "%":
            self._add_token(TokenType.PERCENT, "%", start_line, start_col)
        elif c == "=":
            tt = TokenType.EQ_EQ if self._match("=") else TokenType.EQ
            self._add_token(tt, "==" if tt == TokenType.EQ_EQ else "=", start_line, start_col)
        elif c == "!":
            if self._match("="):
                self._add_token(TokenType.NOT_EQ, "!=", start_line, start_col)
            else:
                raise LexicalError(f"Unexpected character '!'", start_line, start_col, self._filename)
        elif c == "<":
            tt = TokenType.LESS_EQ if self._match("=") else TokenType.LESS
            self._add_token(tt, "<=" if tt == TokenType.LESS_EQ else "<", start_line, start_col)
        elif c == ">":
            tt = TokenType.GREATER_EQ if self._match("=") else TokenType.GREATER
            self._add_token(tt, ">=" if tt == TokenType.GREATER_EQ else ">", start_line, start_col)
        elif c == "(":
            self._paren_depth += 1
            self._add_token(TokenType.LPAREN, "(", start_line, start_col)
        elif c == ")":
            if self._paren_depth > 0:
                self._paren_depth -= 1
            self._add_token(TokenType.RPAREN, ")", start_line, start_col)
        elif c == "{":
            self._paren_depth += 1
            self._add_token(TokenType.LBRACE, "{", start_line, start_col)
        elif c == "}":
            if self._paren_depth > 0:
                self._paren_depth -= 1
            self._add_token(TokenType.RBRACE, "}", start_line, start_col)
        elif c == "[":
            self._paren_depth += 1
            self._add_token(TokenType.LBRACKET, "[", start_line, start_col)
        elif c == "]":
            if self._paren_depth > 0:
                self._paren_depth -= 1
            self._add_token(TokenType.RBRACKET, "]", start_line, start_col)
        elif c == ",":
            self._add_token(TokenType.COMMA, ",", start_line, start_col)
        elif c == ":":
            self._add_token(TokenType.COLON, ":", start_line, start_col)
        elif c == ".":
            self._add_token(TokenType.DOT, ".", start_line, start_col)
        elif c == ";":
            self._add_token(TokenType.SEMICOLON, ";", start_line, start_col)
        elif c == "@":
            self._add_token(TokenType.AT, "@", start_line, start_col)
        else:
            raise LexicalError(f"Illegal character '{c}'", start_line, start_col, self._filename)

    def _handle_indentation(self) -> None:
        spaces = 0
        start_line = self._line
        start_col = self._column

        while not self._is_at_end():
            c = self._peek()
            if c == " ":
                spaces += 1
                self._advance()
            elif c == "\t":
                # Standard Python tab expansion equivalence (8 spaces)
                spaces += 8 - (spaces % 8)
                self._advance()
            elif c == "\n":
                # Blank line, reset column count for line scan and skip
                self._line += 1
                self._column = 1
                self._index += 1
                spaces = 0
                start_line = self._line
                start_col = self._column
            elif c == "#":
                # Comment line, skip until newline
                while not self._is_at_end() and self._peek() != "\n":
                    self._advance()
                if self._peek() == "\n":
                    self._line += 1
                    self._column = 1
                    self._index += 1
                    spaces = 0
                    start_line = self._line
                    start_col = self._column
            else:
                break

        if self._is_at_end():
            return

        # Ignore indentation of blank lines or purely comment lines
        if self._source[self._index] == "\n":
            return

        current_indent = self._indent_stack[-1]

        if spaces > current_indent:
            self._indent_stack.append(spaces)
            self._add_token(TokenType.INDENT, " " * spaces, start_line, start_col)
        elif spaces < current_indent:
            while len(self._indent_stack) > 1 and self._indent_stack[-1] > spaces:
                self._indent_stack.pop()
                self._add_token(TokenType.DEDENT, "", start_line, start_col)
            if self._indent_stack[-1] != spaces:
                raise LexicalError("Unindent does not match any outer indentation level", start_line, start_col, self._filename)