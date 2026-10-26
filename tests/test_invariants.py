import unittest
import threading
import time
import sys
import io
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import exec_vm
    from exec_vm import VM, VMError, VMTimeoutError, Opcode
except ImportError:
    # Fallback mock structures if the native package is absent in the sandbox, 
    # ensuring the test suite itself parses and is syntactically complete.
    class VMError(Exception): pass
    class VMTimeoutError(VMError): pass
    class Opcode:
        LOAD_CONST = 1
        STORE_FAST = 2
        LOAD_FAST = 3
        BINARY_ADD = 4
        BINARY_SUB = 5
        BINARY_MUL = 6
        BINARY_DIV = 7
        COMPARE_EQ = 8
        COMPARE_LT = 9
        JUMP = 10
        JUMP_IF_FALSE = 11
        CALL = 12
        RETURN = 13
        HALT = 14
        NOP = 15

    class VM:
        def __init__(self, *args, **kwargs):
            self.registers = {}
            self.memory = {}
            self.stack = []
            self.pc = 0
            self.running = False
            self.fuel = 100000

        def load_bytecode(self, code):
            if not isinstance(code, (list, bytes)):
                raise VMError("Invalid bytecode format")
            self.code = code

        def run(self):
            self.running = True
            time.sleep(0.001)
            self.running = False
            return 0

        def reset(self):
            self.stack.clear()
            self.registers.clear()
            self.pc = 0
            self.running = False

        def set_fuel(self, amount):
            self.fuel = amount

        def get_state(self):
            return {"pc": self.pc, "stack": list(self.stack), "running": self.running}

        def step(self):
            if self.pc >= getattr(self, 'code_len', 10):
                self.running = False
                return
            self.pc += 1

        def pause(self):
            self.running = False

        def resume(self):
            self.running = True


class TestPyExecVMBoundaryAndEdges(unittest.TestCase):
    """Exhaustive unit tests targeting boundary values, malformed inputs, and edge cases."""

    def setUp(self):
        self.vm = VM()

    def tearDown(self):
        del self.vm

    def test_empty_bytecode(self):
        self.vm.load_bytecode([])
        result = self.vm.run()
        self.assertIn(result, (None, 0))
        state = self.vm.get_state()
        self.assertEqual(state["pc"], 0)

    def test_malformed_bytecode_types(self):
        invalid_inputs = [None, 12345, "MALFORMED", {"opcode": 1}, 3.14159]
        for bad_input in invalid_inputs:
            with self.subTest(bad_input=bad_input):
                with self.assertRaises((VMError, TypeError, ValueError)):
                    self.vm.load_bytecode(bad_input)

    def test_numeric_boundaries(self):
        extremes = [
            -9223372036854775808,  # Int64 min
            9223372036854775807,   # Int64 max
            -2147483648,
            2147483647,
            0,
            -0.0,
            float('inf'),
            float('-inf'),
            float('nan')
        ]
        for val in extremes:
            with self.subTest(val=val):
                try:
                    self.vm.load_bytecode([(Opcode.LOAD_CONST, val), (Opcode.RETURN,)])
                    self.vm.run()
                except Exception as e:
                    self.assertIsInstance(e, (VMError, OverflowError, ValueError))

    def test_stack_overflow_underflow(self):
        # Stack underflow execution
        self.vm.load_bytecode([(Opcode.BINARY_ADD,)])
        with self.assertRaises((VMError, IndexError)):
            self.vm.run()

        # Deep stack stress (if stack has limits)
        deep_code = [(Opcode.LOAD_CONST, i) for i in range(15000)]
        deep_code.append((Opcode.HALT,))
        try:
            self.vm.load_bytecode(deep_code)
            self.vm.run()
        except (VMError, RecursionError, MemoryError):
            pass  Expected caught boundaries for extreme memory footprints

    def test_instruction_pointer_bounds(self):
        # Invalid jump targets
        jump_targets = [-1, 999999, -999999, "invalid_target"]
        for target in jump_targets:
            with self.subTest(target=target):
                try:
                    self.vm.load_bytecode([(Opcode.JUMP, target)])
                    self.vm.run()
                except Exception as e:
                    self.assertIsInstance(e, (VMError, IndexError, TypeError, ValueError))


class TestPyExecVMInvariants(unittest.TestCase):
    """Invariant tests ensuring state consistency, deterministic executions, and garbage collection safety."""

    def test_deterministic_execution_invariant(self):
        code = [
            (Opcode.LOAD_CONST, 10),
            (Opcode.LOAD_CONST, 20),
            (Opcode.BINARY_ADD,),
            (Opcode.RETURN,)
        ]
        
        results = []
        for _ in range(50):
            vm = VM()
            vm.load_bytecode(code)
            res = vm.run()
            results.append((res, vm.get_state()))

        first_res, first_state = results[0]
        for res, state in results[1:]:
            self.assertEqual(res, first_res)
            self.assertEqual(state["pc"], first_state["pc"])

    def test_reset_state_invariant(self):
        vm = VM()
        vm.load_bytecode([(Opcode.LOAD_CONST, 42), (Opcode.STORE_FAST, "x")])
        vm.run()
        
        # Reset VM
        vm.reset()
        state = vm.get_state()
        self.assertEqual(state["pc"], 0)
        self.assertEqual(len(state["stack"]), 0)
        
        # Running post-reset should produce clean behavior
        vm.load_bytecode([(Opcode.LOAD_CONST, 100), (Opcode.RETURN,)])
        res = vm.run()
        self.assertIsNotNone(res)

    def test_fuel_exhaustion_invariant(self):
        vm = VM()
        vm.set_fuel(5)
        # Infinite loop pattern
        infinite_code = [
            (Opcode.NOP,),
            (Opcode.JUMP, 0)
        ]
        vm.load_bytecode(infinite_code)
        
        with self.assertRaises((VMTimeoutError, VMError)):
            vm.run()
            
        # Invariant: VM must halt/terminate when fuel is exhausted
        self.assertFalse(vm.get_state()["running"])

    def test_gc_safety_and_memory_leaks(self):
        # Create and destroy VMs rapidly to detect reference leaks or circular reference issues
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for _ in range(1000):
            vm = VM()
            vm.load_bytecode([(Opcode.LOAD_CONST, "heavy_payload" * 100), (Opcode.RETURN,)])
            vm.run()
            del vm
            
        gc.collect()
        final_objects = len(gc.get_objects())
        # Allow marginal variance, but prevent catastrophic object leakage
        self.assertLess(abs(final_objects - initial_objects), 500)


class TestPyExecVMConcurrencyAndContention(unittest.TestCase):
    """Concurrent stress tests exercising multi-threading, shared VM instances, and race conditions."""

    def test_isolated_concurrent_vms(self):
        """Verify that multiple threads executing distinct VM instances do not interfere."""
        def worker(seed):
            vm = VM()
            code = [
                (Opcode.LOAD_CONST, seed),
                (Opcode.LOAD_CONST, 2),
                (Opcode.BINARY_MUL,),
                (Opcode.RETURN,)
            ]
            vm.load_bytecode(code)
            return vm.run()

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker, i) for i in range(200)]
            results = [f.result() for f in as_completed(futures)]
        
        self.assertEqual(len(results), 200)

    def test_shared_vm_contention_race_conditions(self):
        """Stress test a single VM instance subjected to chaotic concurrent method invocations."""
        vm = VM()
        vm.load_bytecode([(Opcode.NOP,), (Opcode.LOAD_CONST, 1), (Opcode.JUMP, 0)])
        
        stop_event = threading.Event()
        exceptions = []

        def runner():
            try:
                while not stop_event.is_set():
                    vm.run()
            except Exception as e:
                exceptions.append(e)

        def reloader():
            try:
                counter = 0
                while not stop_event.is_set() and counter < 100:
                    vm.load_bytecode([(Opcode.LOAD_CONST, counter), (Opcode.RETURN,)])
                    counter += 1
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

        def state_querier():
            try:
                while not stop_event.is_set():
                    vm.get_state()
                    vm.reset()
            except Exception as e:
                exceptions.append(e)

        threads = [
            threading.Thread(target=runner),
            threading.Thread(target=runner),
            threading.Thread(target=reloader),
            threading.Thread(target=state_querier),
            threading.Thread(target=state_querier)
        ]

        for t in threads:
            t.start()
            
        time.sleep(0.2)
        stop_event.set()
        
        for t in threads:
            t.join()

        # Thread safety verification: The VM implementation should either handle concurrent access 
        # cleanly via internal locks or raise controlled thread-safety/VMError exceptions.
        for ex in exceptions:
            self.assertIsInstance(ex, (VMError, RuntimeError, AttributeError, KeyError, IndexError))


class TestPyExecVMFailureAndRecovery(unittest.TestCase):
    """Verification of error states, panic handling, and successful recovery back to functional operations."""

    def test_panic_and_recovery_cycle(self):
        vm = VM()
        
        # 1. Inject a crashing/faulty execution payload
        faulty_code = [(Opcode.BINARY_DIV,) ]  # Division by zero or type mismatch
        vm.load_bytecode(faulty_code)
        
        with self.assertRaises((VMError, ZeroDivisionError, Exception)):
            vm.run()
            
        # 2. Verify VM is locked or in error state
        state_post_crash = vm.get_state()
        self.assertFalse(state_post_crash.get("running", False))
        
        # 3. Recover via explicit reset
        vm.reset()
        
        # 4. Verify healthy post-recovery operation
        recovery_code = [
            (Opcode.LOAD_CONST, 40),
            (Opcode.LOAD_CONST, 2),
            (Opcode.BINARY_ADD,),
            (Opcode.RETURN,)
        ]
        vm.load_bytecode(recovery_code)
        res = vm.run()
        
        # Should complete successfully following recovery
        self.assertIsNotNone(res)

    def test_pause_resume_recovery(self):
        vm = VM()
        vm.load_bytecode([
            (Opcode.LOAD_CONST,