"""Gate PTX on the saved-thread-index/post-loop-address ptxas hazard.

    python tools/warpdrive/warpdrive_ptx_audit.py agent_space/sass/before

A finding is a candidate for SASS inspection, not proof of a miscompile. Optimized
ptxas can rematerialize base + (tid + C)*scale using a loop-advanced tid register.
This checker follows reaching definitions, including branches and local inline-asm
labels, and reports invariant tid offsets used by memory addresses after a loop
that advances a value from the same tid read. It needs only Python, not a GPU.

The analysis covers explicit add/sub recurrences and direct PTX register dataflow;
it is not a general PTX verifier and does not model calls or implicit state. A clean
result cannot certify arbitrary kernels or assembler versions. See warpdrive's
Dropping down guide for confirmation and workaround requirements. Exit statuses are
0 (no candidates), 1 (candidates), and 2 (invalid input or read failure).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

REGISTER = re.compile(r"%[A-Za-z_$][\w.$]*")
LABEL = re.compile(r"^([\w.$]+):(?!:)")
INSTRUCTION = re.compile(r"^(?:(@!?%?[\w.$]+)\s+)?([a-z][\w.:]*)\s*(.*?)$", re.DOTALL)
NO_DESTINATION = {
    "st",
    "bra",
    "ret",
    "exit",
    "bar",
    "barrier",
    "fence",
    "membar",
    "cp",
    "prefetch",
    "red",
    "trap",
    "brkpt",
    "nanosleep",
    "setmaxnreg",
}


@dataclass(frozen=True)
class Instruction:
    line: int
    opcode: str
    operands: tuple[str, ...]
    predicate: str
    scope: tuple[int, ...]

    @property
    def operation(self) -> str:
        return self.opcode.split(".")[0]

    @property
    def destinations(self) -> tuple[str, ...]:
        if (
            self.operation in NO_DESTINATION
            or not self.operands
            or self.operands[0].lstrip().startswith("[")
        ):
            return ()
        # Vector loads, setp's predicate pair, and shfl's optional predicate result.
        return tuple(REGISTER.findall(self.operands[0]))

    @property
    def sources(self) -> tuple[str, ...]:
        first = 1 if self.destinations else 0
        return tuple(REGISTER.findall(",".join(self.operands[first:])))

    @property
    def addresses(self) -> tuple[str, ...]:
        return tuple(re.findall(r"\[([^]]+)\]", ",".join(self.operands)))


@dataclass(frozen=True)
class Finding:
    kernel: str
    tid_line: int
    offset_line: int
    saved_register: str
    offset: int
    loop_line: int
    increment_line: int
    carried_register: str
    memory_line: int
    memory_opcode: str
    address: str


def parse_ptx(
    text: str,
) -> tuple[list[Instruction], dict[tuple[tuple[int, ...], str], int], dict[int, str]]:
    """Parse instruction statements; qualify reusable inline-asm labels by scope."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), text, flags=re.DOTALL)
    instructions: list[Instruction] = []
    labels: dict[tuple[tuple[int, ...], str], int] = {}
    kernels: dict[int, str] = {}
    scope: list[int] = []
    next_scope = 0
    statement = ""
    start_line = 1
    vector_depth = 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//", 1)[0]
        # NVVM labels occupy their own line, or precede a statement (the small repro).
        if not statement.strip():
            match = LABEL.match(line.strip())
            if match:
                labels[(tuple(scope), match[1])] = len(instructions)
                line = line.strip()[match.end() :]
        for char in line + "\n":
            if not statement.strip() and not char.isspace():
                start_line = line_number
            if char == "{" and (
                not statement.strip()
                or statement.lstrip().startswith((".visible", ".entry", ".func"))
            ):
                entry = re.search(r"\.entry\s+([^\s(]+)", statement)
                if entry:
                    kernels[next_scope] = entry[1]
                scope.append(next_scope)
                next_scope += 1
                statement = ""
            elif char == "{":
                vector_depth += 1
                statement += char
            elif char == "}" and vector_depth:
                vector_depth -= 1
                statement += char
            elif char == "}":
                if not scope or statement.strip():
                    raise ValueError(
                        f"line {line_number}: unmatched scope or incomplete statement"
                    )
                scope.pop()
                statement = ""
            elif char == ";":
                match = INSTRUCTION.match(statement.strip())
                if match:
                    if not scope:
                        raise ValueError(f"line {start_line}: instruction outside a function body")
                    predicate, opcode, operands = match.groups()
                    instructions.append(
                        Instruction(
                            start_line,
                            opcode,
                            tuple(re.split(r",(?![^{}]*\})", operands)),
                            predicate or "",
                            tuple(scope),
                        )
                    )
                statement = ""
            else:
                statement += char
        # Unsuffixed directives are line-based, unlike instruction statements.
        if statement.lstrip().startswith(".") and not statement.lstrip().startswith(
            (".visible", ".entry", ".func")
        ):
            statement = ""
    if scope or vector_depth or statement.strip():
        raise ValueError("unterminated PTX scope or statement")
    nonempty = {instruction.scope[0] for instruction in instructions}
    if not kernels or any(scope_id not in nonempty for scope_id in kernels):
        raise ValueError("expected nonempty PTX kernel entry bodies")
    return instructions, labels, kernels


def audit_ptx(text: str) -> list[Finding]:
    instructions, labels, kernels = parse_ptx(text.rstrip("\x00"))
    predecessors: list[list[int]] = [[] for _ in instructions]
    successors: list[list[int]] = [[] for _ in instructions]
    backward_edges = []
    for index, instruction in enumerate(instructions):
        targets = []
        if instruction.operation == "bra":
            name = instruction.operands[0].strip()
            scope = instruction.scope
            while (scope, name) not in labels and scope:
                scope = scope[:-1]
            if (scope, name) in labels:
                target = labels[(scope, name)]
                targets.append(target)
                if target <= index:
                    backward_edges.append((target, index))
            else:
                raise ValueError(f"line {instruction.line}: unresolved branch label {name}")
        if (
            (instruction.operation not in ("ret", "exit", "bra") or instruction.predicate)
            and index + 1 < len(instructions)
            and instruction.scope[:1] == instructions[index + 1].scope[:1]
        ):
            targets.append(index + 1)
        for target in targets:
            if target < len(instructions):
                predecessors[target].append(index)
                successors[index].append(target)

    @cache
    def definitions(register: str, before: int) -> frozenset[int]:
        found, visited = set(), set()
        pending = deque(predecessors[before])
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            instruction = instructions[index]
            if register in instruction.destinations:
                found.add(index)
                if not instruction.predicate:
                    continue
            pending.extend(predecessors[index])
        return frozenset(found)

    @cache
    def ancestors(register: str, before: int) -> frozenset[int]:
        found, visited = set(), set()
        pending = [(register, before)]
        while pending:
            item = pending.pop()
            if item in visited:
                continue
            visited.add(item)
            for index in definitions(*item):
                found.add(index)
                pending.extend((source, index) for source in instructions[index].sources)
        return frozenset(found)

    def literal(operand: str, before: int) -> int | None:
        try:
            return int(operand.strip(), 0)
        except ValueError:
            defs = (
                definitions(operand.strip(), before) if REGISTER.fullmatch(operand.strip()) else ()
            )
            if len(defs) == 1:
                instruction = instructions[next(iter(defs))]
                if instruction.operation == "mov" and len(instruction.operands) == 2:
                    try:
                        return int(instruction.operands[1].strip(), 0)
                    except ValueError:
                        pass
            return None

    tid_reads = {
        index
        for index, instruction in enumerate(instructions)
        if instruction.operation == "mov"
        and any(re.fullmatch(r"%tid\.[xyz]", source) for source in instruction.sources)
    }
    offsets = []
    for index, instruction in enumerate(instructions):
        if instruction.operation not in ("add", "sub") or len(instruction.operands) != 3:
            continue
        destination, left, right = (operand.strip() for operand in instruction.operands)
        constant = literal(right, index)
        if constant is None and instruction.operation == "add":
            left, right = right, left
            constant = literal(right, index)
        if constant is None or constant == 0 or not REGISTER.fullmatch(left):
            continue
        roots = ancestors(left, index) & tid_reads
        if roots:
            offsets.append(
                (
                    index,
                    destination,
                    left,
                    constant if instruction.operation == "add" else -constant,
                    roots,
                )
            )

    findings = []
    for begin, end in sorted(set(backward_edges)):
        loop = set(range(begin, end + 1))
        after_loop = set()
        pending = [target for index in loop for target in successors[index] if target not in loop]
        while pending:
            index = pending.pop()
            if index in after_loop or index in loop:
                continue
            after_loop.add(index)
            pending.extend(successors[index])
        loop_defs = {
            register
            for instruction in instructions[begin : end + 1]
            for register in instruction.destinations
        }
        increments = []
        for index in range(begin, end + 1):
            instruction = instructions[index]
            if instruction.operation not in ("add", "sub") or len(instruction.operands) != 3:
                continue
            dest, left, right = (operand.strip() for operand in instruction.operands)
            step = literal(right, index)
            if instruction.operation == "add" and step is None:
                left, right = right, left
                step = literal(right, index)
            if dest == left and step not in (None, 0):
                increments.append((index, dest, ancestors(dest, index) & tid_reads))
        for saved, register, source, offset, roots in offsets:
            if saved >= begin or register in loop_defs:
                continue
            # Defining tid and its offset in the same basic block is a verified workaround.
            roots = {
                root
                for root in roots
                if root < saved
                and any(
                    instruction.operation in ("bra", "ret", "exit")
                    for instruction in instructions[root + 1 : saved]
                )
            }
            if not roots:
                continue
            for increment, carried, carried_roots in increments:
                common_roots = roots & carried_roots
                source_registers = {source} | {
                    destination
                    for definition in ancestors(source, saved)
                    for destination in instructions[definition].destinations
                }
                # A fresh IV copied from tid is safe: only recycling a register used
                # to derive the saved offset exposes stale-value rematerialization.
                if not common_roots or carried not in source_registers:
                    continue
                for index in sorted(after_loop):
                    if index <= end:
                        continue
                    instruction = instructions[index]
                    for address in instruction.addresses:
                        if not any(
                            saved in ancestors(addr_register, index)
                            and any(
                                definition > end
                                for definition in definitions(addr_register, index)
                            )
                            for addr_register in REGISTER.findall(address)
                        ):
                            continue
                        # A saved definition reached from another role is not this loop's exit.
                        if instructions[saved].scope[:1] != instruction.scope[:1]:
                            continue
                        line = instruction.line
                        kernel = kernels.get(instruction.scope[0])
                        if kernel is None:
                            continue  # Calls into device functions are outside this analysis.
                        findings.append(
                            Finding(
                                kernel,
                                instructions[min(common_roots)].line,
                                instructions[saved].line,
                                register,
                                offset,
                                instructions[begin].line,
                                instructions[increment].line,
                                carried,
                                line,
                                instruction.opcode,
                                address,
                            )
                        )
    return sorted(
        set(findings),
        key=lambda finding: (finding.memory_line, finding.offset_line, finding.increment_line),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="PTX files or directories (recursive)"
    )
    parser.add_argument("--json", action="store_true", help="emit findings and coverage as JSON")
    args = parser.parse_args()
    files = set()
    for item in args.inputs:
        if not item.exists():
            parser.error(f"input does not exist: {item}")
        files.update(item.rglob("*.ptx") if item.is_dir() else [item])
    if not files:
        parser.error("no PTX files found; an empty inventory is not a passing audit")
    results = {}
    for file in sorted(files):
        try:
            results[str(file)] = audit_ptx(file.read_text())
        except (OSError, ValueError) as error:
            parser.error(f"{file}: {error}")
    if args.json:
        print(
            json.dumps(
                {
                    file: [asdict(finding) for finding in findings]
                    for file, findings in results.items()
                },
                indent=2,
            )
        )
    else:
        for file, findings in results.items():
            print(f"{file}: {len(findings)} candidate(s)")
            for finding in findings:
                print(
                    f"  {finding.kernel}: tid L{finding.tid_line}, {finding.saved_register}=tid{finding.offset:+} L{finding.offset_line}; loop L{finding.loop_line} advances {finding.carried_register} L{finding.increment_line}; {finding.memory_opcode} [{finding.address}] L{finding.memory_line}"
                )
        print(f"Scanned {len(results)} PTX file(s). Candidates require SASS/runtime confirmation.")
    return int(any(results.values()))


if __name__ == "__main__":
    sys.exit(main())
