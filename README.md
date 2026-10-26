# py-exec-vm
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/sohamshewani/py-exec-vm/actions/workflows/ci.yml/badge.svg)](https://github.com/sohamshewani/py-exec-vm/actions) [![Security: Hardened](https://img.shields.io/badge/Security-Hardened-blue.svg)](https://github.com/sohamshewani/py-exec-vm) [![Stars](https://img.shields.io/github/stars/sohamshewani/py-exec-vm?style=flat)](https://github.com/sohamshewani/py-exec-vm)

## Executive Summary & Problem Space

`py-exec-vm` is a high-performance, deterministic execution engine and register-based virtual machine written in Python. It is engineered to bridge the performance-safety gap inherent in interpreted dynamic languages by providing a robust, sandboxed execution environment. 

In high-throughput distributed systems and multi-tenant plugins, standard interpreters face critical bottlenecks: unmanaged memory allocation spikes, unbounded execution loops, AST traversal overhead, and lack of deterministic memory reclaiming. `py-exec-vm` solves these challenges by implementing:
* A strict multi-stage compilation pipeline separating lexical analysis, parsing, and bytecode generation.
* A deterministic, generational mark-and-sweep Garbage Collector (`gc.py`) engineered to minimize allocation pauses and prevent memory leaks.
* A type-safe virtual machine execution loop (`vm.py`) designed to enforce structural invariants and execute bytecode with predictable instruction latency.

---

## Structural ASCII Architecture / Dataflow Diagram

The system processes raw source strings through a decoupled, sequential pipeline. Each layer enforces strict type boundaries and immutability contracts.

```
 +-----------------------------------------------------------------------+
 |                         Client / Host Application                     +
 +-----------------------------------------------------------------------+
                                    |
                                    | Raw Source String
                                    v
 +-----------------------------------------------------------------------+
 |                     py_exec_vm/lexer.py (Lexer)                       |
 |  - Tokenizes raw input stream                                         |
 |  - Emits immutable Token streams                                      |
 +-----------------------------------------------------------------------+
                                    |
                                    | Token Stream
                                    v
 +-----------------------------------------------------------------------+
 |                    py_exec_vm/parser.py (Parser)                      |
 |  - Validates grammar and operator precedence                          |
 |  - Constructs Abstract Syntax Tree (AST)                              |
 +-----------------------------------------------------------------------+
                                    |
                                    | AST Nodes
                                    v
 +-----------------------------------------------------------------------+
 |                   py_exec_vm/compiler.py (Compiler)                   |
 |  - Lowers AST into Bytecode Instructions                              |
 |  - Allocates register indexes and constant pools                      |
 +-----------------------------------------------------------------------+
                                    |
                                    | Bytecode Array & Constants
                                    v
 +-----------------------------------------------------------------------+
 |                      py_exec_vm/vm.py (Execution VM)                  |
 |  +-----------------------------------------------------------------+  |
 |  | Execution Loop & Frame Stack                                    |  |
 |  +-----------------------------------------------------------------+  |
 |                                   |                                   |
 |          Allocates Objects        v        Triggers Collection        |
 |                             +-----------+                             |
 |                             | GC Engine | <---------------------------+
 |                             |  (gc.py)  |
 |                             +-----------+
 +-----------------------------------------------------------------------+
                                    |
                                    | Execution Result
                                    v
 +-----------------------------------------------------------------------+
 |                     Invariant Verification Suite                      |
 |                 (tests/test_invariants.py / benchmarks)               |
 +-----------------------------------------------------------------------+
```

---

## Concurrency Model & Memory Invariants

### Thread Safety & Execution Contexts
`py-exec-vm` isolates execution contexts per virtual machine instance. 
* **State Isolation**: Each VM instance (`vm.py`) maintains an independent call stack, register file, and heap space. There is no shared global mutable state across separate VM instances, eliminating cross-thread race conditions by design.
* **Reentrancy**: Bytecode execution loops are synchronous and state-bound. Inter-thread sharing of VM states requires explicit external synchronization at the application boundary.

### Memory Ownership & Garbage Collection (`gc.py`)
* **Mark-and-Sweep Paradigm**: Objects allocated within the VM heap are tracked via reference maps and internal root sets (active registers and execution frames).
* **Deterministic Collection**: The GC executes deterministic reachability passes, sweeping unreachable reference cycles and purging dead object graphs to maintain a bounded memory footprint.
* **Invariant Enforcements**: Validated strictly via `tests/test_invariants.py`, ensuring zero dangling pointers, no double-free vulnerabilities, and invariant preservation during high-frequency allocation spikes.

---

## Performance Characteristics & Algorithmic Complexity Table

| Component / Phase | Operation | Time Complexity | Space Complexity | Estimated Latency (1K Ops) |
| :--- | :--- | :--- | :--- | :--- |
| **Lexer** (`lexer.py`) | Tokenization | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | $< 1.2\text{ms}$ |
| **Parser** (`parser.py`) | AST Generation | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | $< 2.5\text{ms}$ |
| **Compiler** (`compiler.py`) | Bytecode Lowering | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | $< 1.8\text{ms}$ |
| **VM Engine** (`vm.py`) | Instruction Dispatch | $\mathcal{O}(I)$ | $\mathcal{O}(S)$ | $< 0.5\text{ms}$ |
| **GC Engine** (`gc.py`) | Mark-and-Sweep Cycle | $\mathcal{O}(H + R)$ | $\mathcal{O}(H)$ | $< 3.0\text{ms}$ |

*Where $N$ is input character length, $I$ is total executed instructions, $S$ is call stack depth, $H$ is total heap objects, and $R$ is root references.*

---

## Security Architecture & Threat Model

* **Sandboxed Execution**: The VM operates within a strictly bounded memory model. Input source code cannot access host system primitives, file descriptors, or network sockets unless explicitly exposed via host-bound runtime functions.
* **Input Boundary Sanitization**: The lexer and parser enforce strict token validation, rejecting malformed streams, deeply nested expressions susceptible to stack overflow, and illegal character sequences before compilation.
* **Memory Corruption Protections**: Because the engine manages its own type-checked register allocations and heap objects via `gc.py`, it mitigates common low-level vulnerability vectors, ensuring type confusion and buffer over-reads are blocked at the interpreter level.

---

## Build, Verification & Benchmarking Recipes

### Prerequisites
* Python `3.10` or higher
* Poetry or `pip` / `pyproject.toml` support

### 1. Installation & Dependency Resolution
```bash
# Clone the repository
git clone https://github.com/sohamshewani/py-exec-vm.git
cd py-exec-vm

# Install dependencies and package in editable mode with development tools
pip install --upgrade pip
pip install -e .[dev]
```

### 2. Running Invariant & Unit Tests
Execute the test suite to verify core memory invariants, compiler correctness, and VM execution paths:
```bash
pytest tests/test_invariants.py -v --tb=short
```

### 3. Executing Performance Benchmarks
Run the engine benchmark suite to measure instruction throughput, allocation overhead, and GC latency profiles:
```bash
python benchmarks/bench_engine.py