from __future__ annotations

import sys
import time
from typing import List, Set, Optional, Iterator, Any, Dict

from py_exec_vm.object_model import HeapObject, GCHeader, VMObject, GCColor, VmTypeDescriptor


class GarbageCollector:
    """
    Stop-the-world Mark-and-Sweep Garbage Collector utilizing a tri-color marking
    abstraction (White, Grey, Black) with an explicit worklist stack to prevent
    recursive Python frame allocation limits during deep graph sweeps.
    Integrated with arena-based chunk allocation tracking and allocation quotas.
    """

    def __init__(self, allocation_threshold: int = 1024 * 1024) -> None:
        self.heap_head: Optional[HeapObject] = None
        self.heap_tail: Optional[HeapObject] = None
        self.total_allocated_bytes: int = 0
        self.allocation_threshold: int = allocation_threshold
        self.collected_count: int = 0
        self.total_pause_time_ns: int = 0
        self._allocation_count_since_gc: int = 0
        self._gc_trigger_interval: int = 4096

    def register_allocation(self, obj: HeapObject) -> None:
        """
        Registers a newly allocated HeapObject into the intrusive doubly-linked list
        managed by the garbage collector and checks allocation quotas.
        """
        size_bytes = obj.size_in_bytes()
        self.total_allocated_bytes += size_bytes
        self._allocation_count_since_gc += 1

        # Intrusive doubly-linked list insertion at the head
        if self.heap_head is None:
            self.heap_head = obj
            self.heap_tail = obj
            obj.gc_prev = None
            obj.gc_next = None
        else:
            obj.gc_next = self.heap_head
            obj.gc_prev = None
            self.heap_head.gc_prev = obj
            self.heap_head = obj

        # Check if we need to trigger a cooperative GC safepoint
        if (
            self.total_allocated_bytes >= self.allocation_threshold
            or self._allocation_count_since_gc >= self._gc_trigger_interval
        ):
            # The execution engine will poll safepoints, but we flag the requirement.
            pass

    def unregister_allocation(self, obj: HeapObject) -> None:
        """
        Removes an object from the intrusive heap list during sweeping or explicit cleanup.
        """
        if obj.gc_prev is not None:
            obj.gc_prev.gc_next = obj.gc_next
        else:
            self.heap_head = obj.gc_next

        if obj.gc_next is not None:
            obj.gc_next.gc_prev = obj.gc_prev
        else:
            self.heap_tail = obj.gc_prev

        obj.gc_prev = None
        obj.gc_next = None
        self.total_allocated_bytes -= obj.size_in_bytes()

    def collect(self, roots: List[HeapObject]) -> int:
        """
        Executes a full Stop-The-World Mark-and-Sweep collection cycle.
        Returns the number of bytes reclaimed.
        """
        start_time = time.perf_counter_ns()
        
        # Phase 1: Mark
        self._mark_phase(roots)

        # Phase 2: Sweep
        reclaimed_bytes = self._sweep_phase()

        end_time = time.perf_counter_ns()
        pause_duration = end_time - start_time
        self.total_pause_time_ns += pause_duration
        self.collected_count += 1
        self._allocation_count_since_gc = 0

        return reclaimed_bytes

    def _mark_phase(self, roots: List[HeapObject]) -> None:
        """
        Tri-color marking phase using an explicit worklist stack (Grey set)
        to prevent recursion limits.
        """
        worklist: List[HeapObject] = []

        # Initialize all heap objects to WHITE
        current = self.heap_head
        while current is not None:
            current.header.color = GCColor.WHITE
            current = current.gc_next

        # Process roots: transition from WHITE to GREY, push to worklist
        for root in roots:
            if root is not None and isinstance(root, HeapObject):
                if root.header.color == GCColor.WHITE:
                    root.header.color = GCColor.GREY
                    worklist.append(root)

        # Explicit worklist traversal (DFS/BFS via stack)
        while len(worklist) > 0:
            obj = worklist.pop()
            obj.header.color = GCColor.BLACK

            # Inspect references/children of the object
            for child in obj.iter_references():
                if child is not None and isinstance(child, HeapObject):
                    if child.header.color == GCColor.WHITE:
                        child.header.color = GCColor.GREY
                        worklist.append(child)

    def _sweep_phase(self) -> int:
        """
        Sweeps the intrusive heap list, reclaiming WHITE objects and resetting
        BLACK objects back to WHITE for the next collection cycle.
        """
        reclaimed_bytes = 0
        current = self.heap_head

        while current is not None:
            nxt = current.gc_next
            if current.header.color == GCColor.BLACK:
                # Reset to WHITE for next GC cycle
                current.header.color = GCColor.WHITE
                current = nxt
            elif current.header.color == GCColor.WHITE:
                # Unreachable object: reclaim
                size = current.size_in_bytes()
                self.unregister_allocation(current)
                current.destroy()
                reclaimed_bytes += size
                current = nxt
            else:
                # Should not encounter GREY during sweep phase
                current.header.color = GCColor.WHITE
                current = nxt

        # Adjust threshold dynamically based on live objects ratio
        self.allocation_threshold = max(
            1024 * 1024, (self.total_allocated_bytes * 2)
        )
        return reclaimed_bytes

    def get_telemetry_metrics(self) -> Dict[str, Any]:
        """
        Returns built-in performance counters for GC telemetry.
        """
        return {
            "total_allocated_bytes": self.total_allocated_bytes,
            "allocation_threshold": self.allocation_threshold,
            "collected_count": self.collected_count,
            "total_pause_time_ns": self.total_pause_time_ns,
            "average_pause_time_ns": (
                self.total_pause_time_ns // self.collected_count
                if self.collected_count > 0
                else 0
            ),
        }