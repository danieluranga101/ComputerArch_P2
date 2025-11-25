# 5-Stage Pipelined CPU Simulator

This Python program simulates a 5-stage pipelined CPU architecture with an 8-bit accumulator model. It implements the pipeline stages — Instruction Fetch (IF), Instruction Decode (ID), Execute (EX), and Memory Access/Write-Back (MEM/WB) — without hazard detection or forwarding.

## CPU Architecture

### Registers

*   **ACC:** 8-bit accumulator
*   **Z:** Zero flag
*   **N:** Negative flag
*   **PC:** 8-bit Program Counter

### Memory

*   256 bytes of Main Memory for data.
*   A separate instruction memory to hold the program.

### Instruction Set

The simulator supports the following opcodes:

*   **Data Transfer:**
    *   `LOAD addr`: Load from memory direct. `ACC <- MEM[addr]`
    *   `STORE addr`: Store to memory direct. `MEM[addr] <- ACC`
    *   `LOADI addr`: Load from memory indirect. `ACC <- MEM[MEM[addr]]`
    *   `STOREI addr`: Store to memory indirect. `MEM[MEM[addr]] <- ACC`
*   **Arithmetic/Logic:**
    *   `ADD addr`: `ACC <- ACC + MEM[addr]`
    *   `SUB addr`: `ACC <- ACC - MEM[addr]`
    *   `AND addr`: `ACC <- ACC & MEM[addr]`
    *   `OR addr`: `ACC <- ACC | MEM[addr]`
*   **Control Flow:**
    *   `JMP addr`: Unconditional jump. `PC <- addr`
    *   `JZ addr`: Jump if Zero flag is set. `PC <- addr if Z == 1`
    *   `JN addr`: Jump if Negative flag is set. `PC <- addr if N == 1`
*   **Other:**
    *   `NOP`: No operation.
    *   `HALT`: Stops the simulation.

## Pipeline Stages

The simulation is based on a classic 5-stage RISC pipeline:

1.  **IF (Instruction Fetch):** Fetches the next instruction from instruction memory.
2.  **ID (Instruction Decode):** Decodes the instruction, reads registers, and generates control signals.
3.  **EX (Execute):** Performs ALU operations or calculates addresses.
4.  **MEM (Memory Access):** Reads from or writes to data memory.
5.  **WB (Write-Back):** Writes the result back to the accumulator.

**Note:** This is a simplified model and does **not** implement hazard detection or data forwarding. NOPs are required to avoid data hazards.

## How to Run

The script can be executed directly from the command line with the following options:
    First required argument is always input file
    -n : if present then it runs the non pipelined version of ISA (used for benchmark comparison)
    -o <output.mem> : if present then writes final memory state to output.mem
    -v : enables verbose debugging prints

```bash
python .\s12_pipeline.py .\Sort_genSize_bubbleTime_v2.mem -o bubble_pipe_result.mem
```

## Branch Prediction
TODO:
1. Static branch prediction:
    always take
    never take
    take forward
    take backwards

2. Dynamic branch prediction
    counter
    last taken

The output will show a cycle-by-cycle trace of the pipeline registers and the final state of the CPU and memory after the test program has run.

## Demo Program
If no file is given for input then it will run a small demo program
The script includes a self-contained test program under the `if __name__ == "__main__":` block. This program is designed to exercise various instructions and verify the correct behavior of the pipeline, including direct and indirect memory operations, arithmetic, and conditional branching.
