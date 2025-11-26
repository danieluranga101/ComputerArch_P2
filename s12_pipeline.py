''' Description:
This Python program simulates a 5-stage pipelined CPU architecture with an 8-bit accumulator 
model. It implements the pipeline stages — Instruction Fetch (IF), Instruction Decode (ID),
Execute (EX), and Memory Access/Write-Back (MEM/WB) — without hazard detection or forwarding.

Sections Overview:
1. Instruction Class Definition:
   Defines the Instruction structure with opcode and operand fields.

2. Processor Architecture State Initialization:
   Declares and initializes CPU registers (ACC, flags Z and N), program counter (PC),
   and both instruction and data memory.

3. Helper Functions:
   - mask8(): Ensures 8-bit arithmetic by masking values.
   - set_flags_from(): Updates zero (Z) and negative (N) flags based on accumulator output.

4. Pipeline Stage Implementations:  opcode (LOAD,STORE.ADD,SUB,AND,OR,LOADI,STOREI,JMP,JN,JZ,HALT)
   - fetch_instruction(): Fetches instructions into IF/ID pipeline register.
   - decode_instruction(): Decodes opcode, extracts operands, and sets control signals.
   - execute_instruction(): Performs arithmetic/logic operations or branch resolution.
   - memory_access(): Handles memory read/write for direct and indirect addressing modes.
   - write_back(): Updates the accumulator and status flags based on results.

5. run function:
   Simulates the pipeline execution cycle-by-cycle, passing data through the pipeline 
   registers, updating state, and generating a trace of CPU activity.

6. Demo Program :
   Demonstrates indirect and direct memory operations using LOADI, STOREI, ADD, AND, LOAD, HALT, JN
   instructions. Memory is preloaded to verify pointer-based behavior.'''

import sys
from typing import List, Dict, Any
from process_mem import Instruction, mask8, to_signed8, process_mem_file, write_mem_file

DEBUG = False
def printg(msg: str):
    if DEBUG:
        print(msg)

# Processor Architecture State initilization (Registers, Flags, and Memory)
class S12PipelineCPU:
    def __init__(self, mem):
        self.PC = 0        # 8-bit Program Counter
        self.ACC = 0       # 8-bit accumulator reg, always stored as signed
        self.Z = 0         # zero flag
        self.N = 0         # negative flag (bit7)
        self.HALT = False  # halt flag
        self.MEM: Dict[int, int] = {i: 0 for i in range(256)}  # Main Memory
        if mem:
            for k, v in initial_mem.items():
                self.MEM[k & 0xFF] = v & 0xFFF
        # Pipeline registers
        self.IF_ID = {}
        self.ID_EX = {}
        self.EX_MEM = {}
        self.MEM_WB = {}
        # Snapshots of previous cycle pipeline registers for hazard detection
        # and true single cycle simulation
        self.prevIF_ID = {}
        self.prevID_EX = {}
        self.prevEX_MEM = {}
        self.prevMEM_WB = {}
        self.perf_counters = {
            "cycles": 0,
            "instructions": 0,
            "stalls": 0,
            "flushes": 0,
            "forwards": {
                "EX_MEM": 0,
                "MEM_WB": 0
            },
            "instruction_mix": {
                "LOAD": 0,
                "STORE": 0,
                "LOADI": 0,
                "STOREI": 0,
                "ADD": 0,
                "SUB": 0,
                "AND": 0,
                "OR": 0,
                "JMP": 0,
                "JN": 0,
                "JZ": 0,
                "HALT": 0,
            }
        }
        self.branch_prediction = {
            "method": "none",       # default prediction method
            "last_PC": 0,
            "jumped": False,
            "taken": 0,
            "not_taken": 0,
            "correct": 0,
            "incorr": 0,

            # Confusion Matrix
            "TP": 0,   # predicted taken, actual taken
            "FP": 0,   # predicted taken, actual not taken
            "FN": 0,   # predicted not taken, actual taken
            "TN": 0    # predicted not taken, actual not taken
        }

    
    def display_performance_counters(self):
        print("Performance Counters:")
        print(f"Total Cycles: {self.perf_counters['cycles']}")
        print(f"Total Instructions Executed: {self.perf_counters['instructions']}")
        print(f"Total Stalls: {self.perf_counters['stalls']}")
        print(f"Total Flushes: {self.perf_counters['flushes']}")
        print("Data Forwards:")
        for stage, count in self.perf_counters['forwards'].items():
            print(f"  From {stage}: {count}")
        print("Instruction Mix:")
        for instr, count in self.perf_counters['instruction_mix'].items():
            print(f"  {instr}: {count}")

    def display_branch_prediction_metrics(self):
        print("Branch Prediction Metrics:")
        print(f"  Method: {self.branch_prediction['method']}")
        print(f"  Branches Taken: {self.branch_prediction['taken']}")
        print(f"  Branches Not Taken: {self.branch_prediction['not_taken']}")
        print(f"  Correct Predictions: {self.branch_prediction['correct']}")
        print(f"  Incorrect Predictions: {self.branch_prediction['incorr']}")
        print(f"  Confusion Matrix:")
        print(f"                 Actual Taken | Actual Not Taken")
        print(f"  Pred Taken      {self.branch_prediction['TP']:6}        {self.branch_prediction['FP']:6}")
        print(f"  Pred Not Taken  {self.branch_prediction['FN']:6}        {self.branch_prediction['TN']:6}")


    def get_forwarded_acc(self):
        """Get ACC value with forwarding from EX_MEM or MEM_WB if available"""
        acc_value = self.ACC
        # Check EX_MEM for ALU result
        option = -1 # no forwarding
        if self.prevEX_MEM.get("valid", False) and self.prevEX_MEM.get("opcode") in (
            Instruction.opcode_to_int("ADD"), Instruction.opcode_to_int("SUB"),
            Instruction.opcode_to_int("AND"), Instruction.opcode_to_int("OR")):
            acc_value = self.prevEX_MEM.get("alu_out", acc_value)
            self.perf_counters['forwards']['EX_MEM'] += 1
            option = 0
        
        # Check MEM_WB for ALU result last because it takes priority
        if self.prevMEM_WB.get("valid", False) and self.prevMEM_WB.get("opcode") in (
            Instruction.opcode_to_int("LOAD"), Instruction.opcode_to_int("LOADI")):
            acc_value = self.prevMEM_WB.get("data_mem", acc_value)
            self.perf_counters['forwards']['MEM_WB'] += 1
            option = 1
        if option == 0:
            printg(f"Forwarded ACC from EX_MEM used. val: {acc_value}")
        elif option == 1:
            printg(f"Forwarded ACC from MEM_WB used. val: {acc_value}")
        return acc_value
    
    def detect_hazard(self):
        """If current execution stage is LOAD and next exectute stage is ALU op,
           or if current executions stage is STORE and next execute stage is ALU op, stall"""
        # Load use hazard
        if not (self.prevID_EX.get("valid", False) and self.prevIF_ID.get("valid", False)):
            return False
        execute_op = Instruction(self.prevID_EX.get("opcode", None), 0)
        printg(f"Detecting hazards with EX opcode: {execute_op.opcode_to_string()}")
        if execute_op.opcode in (Instruction.opcode_to_int("LOAD"), Instruction.opcode_to_int("LOADI")):
            decode_opcode = Instruction.binary_to_instruction(self.prevIF_ID.get("instr", 0))
            printg(f"Checking hazard with ID opcode: {decode_opcode.opcode_to_string()}")
            if decode_opcode.opcode in (Instruction.opcode_to_int("ADD"),
                                        Instruction.opcode_to_int("SUB"),
                                        Instruction.opcode_to_int("AND"),
                                        Instruction.opcode_to_int("OR"),
                                        Instruction.opcode_to_int("JN"),
                                        Instruction.opcode_to_int("JZ"),
                                        Instruction.opcode_to_int("JMP")):
                printg(f"Hazard detected between ID_EX operand {execute_op.opcode} and IF_ID operand {decode_opcode.opcode_to_string()}")
                return True
        # Read after write hazard
        if execute_op.opcode in (Instruction.opcode_to_int("STORE"), Instruction.opcode_to_int("STOREI")):
            decode_opcode = Instruction.binary_to_instruction(self.prevIF_ID.get("instr", 0))
            printg(f"Checking hazard with ID opcode: {decode_opcode.opcode_to_string()}")
            if decode_opcode.opcode in (Instruction.opcode_to_int("ADD"),
                                        Instruction.opcode_to_int("SUB"),
                                        Instruction.opcode_to_int("AND"),
                                        Instruction.opcode_to_int("OR"),
                                        Instruction.opcode_to_int("JN"),
                                        Instruction.opcode_to_int("JZ"),
                                        Instruction.opcode_to_int("JMP")):
                # For store we can check if the opcodes are the same. We only have a hazard if they are different
                if execute_op.opcode == Instruction.opcode_to_int("STORE") and decode_opcode.operand != execute_op.operand:
                    return False
                printg(f"Hazard detected between ID_EX operand {execute_op.opcode} and IF_ID operand {decode_opcode.opcode_to_string()}")
                return True
        return False
    
    def flush_for_control_hazard(self, halt=False):
        """Flush instructions in IF_ID and ID_EX due to control hazard"""
        printg("Flushing pipeline due to control hazard")
        self.IF_ID = {"valid": False}
        self.ID_EX = {"valid": False}
        if not halt:
            # only count flush if not halting
            self.perf_counters['flushes'] += 1

    def fetch_instruction(self):
        """IF: fetch instruction from memory into IF_ID"""
        if self.HALT:
            self.IF_ID = {"valid": False}
            return
        
        if 0 <= self.PC < 0xFF:
            instruction = self.MEM[self.PC]
            printg(f"IF: Fetched instruction {instruction:012b} from MEM[{self.PC}]")
            self.IF_ID = {
                "instr": instruction,  # instr[11:0] (object)
                "valid": True
            }
            instruction_obj = Instruction.binary_to_instruction(instruction)
            if instruction_obj.opcode in (Instruction.opcode_to_int("JMP"),
                                          Instruction.opcode_to_int("JN"),
                                          Instruction.opcode_to_int("JZ")):
                printg(f"IF: Branch instruction detected: {instruction_obj.opcode_to_string()}")
                self.branch_prediction["last_PC"] = self.PC
                match self.branch_prediction["method"]:
                    case "static_not_taken":
                        self.PC+=1
                    case "static_taken":
                        self.PC = instruction_obj.operand & 0xFF
                        self.branch_prediction["jumped"] = True
                    case "none":
                        self.PC+=1
                    case _:
                        print("Unknown branch prediction method")
                        pass
            else:
                self.PC+=1

    def decode_instruction(self):
        """ID: decode instruction, extract operands"""
        if self.HALT:
            self.ID_EX = {"valid": False}
            return
        if self.IF_ID is None or not self.IF_ID.get("valid", False) or self.IF_ID["instr"] is None:
            self.ID_EX = {"valid": False}
            return
        
        instruction = Instruction.binary_to_instruction(self.IF_ID["instr"])
        instruction_str = instruction.opcode_to_string().upper()
        printg(f"ID: Opcode: {instruction_str}, Operand: {instruction.operand}")
        self.ID_EX = {"opcode": instruction.opcode, "operand": instruction.operand,
                      "valid": True}

    def alu(self, op, b):
        """Simple ALU operations"""
        a = self.get_forwarded_acc() # accumulator will always be signed for now
        b = to_signed8(mask8(b))
        if op == Instruction.opcode_to_int("ADD"):
            result = to_signed8(a + b)
        elif op == Instruction.opcode_to_int("SUB"):
            result = to_signed8(a - b)
        elif op == Instruction.opcode_to_int("AND"):
            result = to_signed8(a & b)
        elif op == Instruction.opcode_to_int("OR"):
            result = to_signed8(a | b)
        else:
            result =  to_signed8(b)  # passthrough operand
        op_obj = Instruction(op, 0)
        printg(f"ALU Operation: {op_obj.opcode_to_string()}, A: {a}, B: {b} => Result: {to_signed8(mask8(result))} binary: {mask8(result):08b}")

        return mask8(result)

        
    def execute_instruction(self):
        """EX: do ALU ops or branch resuoltuion"""
        if self.HALT:
            self.EX_MEM = {"valid": False}
            return
        if self.ID_EX is None or not self.ID_EX.get("valid", False):
            self.EX_MEM = {"valid": False}
            return
        
        instruction = Instruction(self.ID_EX["opcode"], self.ID_EX["operand"])
        instruction_str = instruction.opcode_to_string().upper()
        printg(f"EX: Opcode: {instruction_str}, Operand: {instruction.operand}")

        if instruction.opcode in (Instruction.opcode_to_int("ADD"), Instruction.opcode_to_int("SUB"),
                                  Instruction.opcode_to_int("AND"), Instruction.opcode_to_int("OR")):
            # ALU operation with MEM[operand]
            alu_result = self.alu(instruction.opcode, self.MEM.get(instruction.operand, 0))
            self.EX_MEM = {"alu_out": alu_result, "opcode": instruction.opcode,
                           "operand": instruction.operand, "valid": True}
            self.perf_counters['instruction_mix'][instruction_str] += 1
        elif instruction.opcode is Instruction.opcode_to_int("JMP"):
            # Unconditional jump
            if(self.branch_prediction["method"] == "none"):
                self.PC = instruction.operand & 0xFF
                self.flush_for_control_hazard()
            else:
                if (self.branch_prediction["jumped"]):
                    self.branch_prediction["correct"] += 1
                    self.branch_prediction["TP"] += 1
                else:
                    self.branch_prediction["incorr"] += 1
                    self.branch_prediction["FN"] += 1
                    self.PC = instruction.operand & 0xFF
                    self.flush_for_control_hazard()
            self.branch_prediction["taken"] += 1
            self.EX_MEM = {"opcode": instruction.opcode, "operand": instruction.operand, "valid": True}
            self.perf_counters['instruction_mix']['JMP'] += 1
        elif instruction.opcode is Instruction.opcode_to_int("JN"):
            # Jump if negative
            forwarded_acc = to_signed8(self.get_forwarded_acc())
            printg(f"JN: Checking N flag with forwarded ACC value: {forwarded_acc}")
            if forwarded_acc < 0:
                if(self.branch_prediction["method"] == "none"):
                        self.PC = instruction.operand & 0xFF
                        self.flush_for_control_hazard()
                else:
                    if (self.branch_prediction["jumped"]):
                        self.branch_prediction["correct"] += 1
                        self.branch_prediction["TP"] += 1
                    else:
                        self.branch_prediction["incorr"] += 1
                        self.branch_prediction["FN"] += 1
                        self.PC = instruction.operand & 0xFF
                        self.flush_for_control_hazard()
                self.branch_prediction["taken"] += 1
            else:
                if (self.branch_prediction["jumped"]):
                    self.branch_prediction["incorr"] += 1
                    self.branch_prediction["FP"] += 1
                    self.PC = self.branch_prediction.get("last_PC", self.PC) + 1
                    self.flush_for_control_hazard()
                else:
                    self.branch_prediction["correct"] += 1
                    self.branch_prediction["TN"] += 1
                self.branch_prediction["not_taken"] += 1
            self.EX_MEM = {"opcode": instruction.opcode, "operand": instruction.operand, "valid": True}
            self.perf_counters['instruction_mix']['JN'] += 1
        elif instruction.opcode is Instruction.opcode_to_int("JZ"):
            # Jump if zero
            forwarded_acc = to_signed8(self.get_forwarded_acc())
            if forwarded_acc == 0:
                if(self.branch_prediction["method"] == "none"):
                    self.PC = instruction.operand & 0xFF
                    self.flush_for_control_hazard()
                else:
                    if (self.branch_prediction["jumped"]):
                        self.branch_prediction["correct"] += 1
                        self.branch_prediction["TP"] += 1
                    else:
                        self.branch_prediction["incorr"] += 1
                        self.branch_prediction["FN"] += 1
                        self.PC = instruction.operand & 0xFF
                        self.flush_for_control_hazard()
                    self.branch_prediction["taken"] += 1
            else:
                if (self.branch_prediction["jumped"]):
                    self.branch_prediction["incorr"] += 1
                    self.branch_prediction["FP"] += 1
                    self.PC = self.branch_prediction.get("last_PC", self.PC) + 1
                    self.flush_for_control_hazard()
                else:
                    self.branch_prediction["correct"] += 1
                    self.branch_prediction["TN"] += 1
                self.branch_prediction["not_taken"] += 1
            self.EX_MEM = {"opcode": instruction.opcode, "operand": instruction.operand, "valid": True}
            self.perf_counters['instruction_mix']['JZ'] += 1
        elif instruction.opcode is Instruction.opcode_to_int("HALT"):
            self.HALT = True
            self.EX_MEM = {"opcode": instruction.opcode, "operand": instruction.operand, "valid": True}
            self.flush_for_control_hazard(True) # don't count halt flush for performance coutners
            self.perf_counters['instruction_mix']['HALT'] += 1
        else:
            # Other instructions (LOAD, STORE, LOADI, STOREI) pass through
            self.EX_MEM = {"opcode": instruction.opcode, "operand": instruction.operand, "valid": True}
            self.perf_counters['instruction_mix'][instruction_str] += 1
        self.perf_counters['instructions'] += 1
    
    def memory_access(self):
        """MEM: memory access"""
        if self.EX_MEM is None or not self.EX_MEM.get("valid", False):
            self.MEM_WB = {"valid": False}
            return
        # data_mem is the value read from memory (for LOAD/LOADI)
        # alu_out is the ALU result (for ALU ops)
        # in write-back we will write to ACC from data_mem or alu_out as needed
        if self.EX_MEM.get("opcode", None) is Instruction.opcode_to_int("LOAD"):
            # Direct load
            data = self.MEM.get(self.EX_MEM["operand"], 0)
            printg(f"MEM: LOAD: Loaded value {data} from MEM[{self.EX_MEM['operand']}]")
            self.MEM_WB = {"data_mem": data, "opcode": self.EX_MEM["opcode"],
                           "operand": self.EX_MEM["operand"], "valid": True}
        elif self.EX_MEM.get("opcode", None) is Instruction.opcode_to_int("STORE"):
            # Direct store
            self.MEM[self.EX_MEM["operand"]] = self.ACC & 0xFF
            printg(f"MEM: STORE: Stored ACC value {self.ACC & 0xFF} to MEM[{self.EX_MEM['operand']}]")
            self.MEM_WB = {"opcode": self.EX_MEM["opcode"],
                           "operand": self.EX_MEM["operand"], "valid": True}
        elif self.EX_MEM.get("opcode", None) is Instruction.opcode_to_int("LOADI"):
            # Indirect load
            ptr = self.MEM.get(self.EX_MEM["operand"], 0) & 0xFF
            data = self.MEM.get(ptr, 0)
            printg(f"MEM: LOADI: Loaded value {data} from MEM[MEM[{self.EX_MEM['operand']}]] = MEM[{ptr}]")
            self.MEM_WB = {"data_mem": data, "opcode": self.EX_MEM["opcode"],
                           "operand": self.EX_MEM["operand"], "valid": True}
        elif self.EX_MEM.get("opcode", None) is Instruction.opcode_to_int("STOREI"):
            # Indirect store
            ptr = self.MEM.get(self.EX_MEM["operand"], 0) & 0xFF
            self.MEM[ptr] = self.ACC & 0xFF
            printg(f"MEM: STOREI: Stored ACC value {self.ACC & 0xFF} to MEM[MEM[{self.EX_MEM['operand']}]] = MEM[{ptr}]")
            self.MEM_WB = {"opcode": self.EX_MEM["opcode"],
                           "operand": self.EX_MEM["operand"], "valid": True}
        else:
            # ALU ops and others pass through
            printg(f"MEM: passthrough through ALU result {self.EX_MEM.get('alu_out', None)}")
            self.MEM_WB = {"alu_out": self.EX_MEM.get("alu_out", None), # if not present then its a jump/halt
                           "opcode": self.EX_MEM["opcode"],
                           "operand": self.EX_MEM["operand"], "valid": True}
                      
    def write_back(self):
        """WB: write back to ACC"""
        if self.MEM_WB is None or not self.MEM_WB.get("valid", False):
            return
        
        # write back to ACC based on opcode, use data from Load or ALU result
        if self.MEM_WB.get("opcode", None) is Instruction.opcode_to_int("LOAD"):
            self.ACC = to_signed8(mask8(self.MEM_WB["data_mem"]))
            printg(f"WB: LOAD: ACC updated to {self.ACC}")
        elif self.MEM_WB.get("opcode", None) is Instruction.opcode_to_int("LOADI"):
            self.ACC = to_signed8(mask8(self.MEM_WB["data_mem"]))
            printg(f"WB: LOADI: ACC updated to {self.ACC}")
        elif self.MEM_WB.get("opcode", None) in (Instruction.opcode_to_int("ADD"),
                                                 Instruction.opcode_to_int("SUB"),
                                                 Instruction.opcode_to_int("AND"),
                                                 Instruction.opcode_to_int("OR")):
            printg(f"WB: ALU op: ACC updated to {self.MEM_WB['alu_out']}")
            self.ACC = to_signed8(mask8(self.MEM_WB["alu_out"]))
        
        # Update flags
        self.Z = 1 if self.ACC == 0 else 0
        self.N = 1 if self.ACC < 0 else 0
        printg(f"WB: Flags updated: Z={self.Z}, N={self.N}")

    def simulate_pipelined_cycle(self):
        printg(f"{self.PC}")
        # Save current pipeline registers for next cycle hazard detection and forwarding
        self.prevIF_ID = self.IF_ID.copy()
        self.prevID_EX = self.ID_EX.copy()
        self.prevEX_MEM = self.EX_MEM.copy()
        self.prevMEM_WB = self.MEM_WB.copy()

        self.write_back()
        self.memory_access()
        self.execute_instruction()

        if self.detect_hazard():
            self.perf_counters['stalls'] += 1
            # Insert a stall by keeping ID_EX the same
            # rerun IF and ID stages
            self.ID_EX = {"valid": False}
            printg("\n")
            return

        self.decode_instruction()
        self.fetch_instruction()
        printg("\n")

    def simulate_non_pipelined_5cycle(self):
        printg(f"{self.PC}")
        # Save current pipeline registers for next cycle hazard detection and forwarding
        self.prevIF_ID = self.IF_ID.copy()
        self.prevID_EX = self.ID_EX.copy()
        self.prevEX_MEM = self.EX_MEM.copy()
        self.prevMEM_WB = self.MEM_WB.copy()
        self.fetch_instruction()
        self.decode_instruction()
        self.execute_instruction()
        self.memory_access()
        self.write_back()
        printg("\n")


    def run(self, max_cycles=100, start_pc=0, pipelined=False):
            self.perf_counters['cycles'] = 0
            halted = False
            while self.perf_counters['cycles'] < max_cycles and not halted:
                if not pipelined:
                    self.simulate_non_pipelined_5cycle()
                    self.perf_counters['cycles'] += 5
                else:
                    self.simulate_pipelined_cycle()
                    self.perf_counters['cycles'] += 1
                if self.EX_MEM.get('valid', False) is False and self.MEM_WB.get("valid", False) is False and self.HALT:
                    halted = True
            return self.perf_counters['cycles']
#-----------------------------------------------------------------------------------------------------------------------------------------
# Program to verify implementation

if __name__ == "__main__":
    """
    Program to test:
    LOADI, ADD, STOREI, LOAD, STORE, AND, OR, SUB, JN, HALT

    Memory Setup:
      MEM[40]  = 41     (pointer for LOADI)
      MEM[41] = 50     (value for LOADI’s indirect read and for ADD via M[10])
      MEM[42] = 43     (pointer for STOREI)
      MEM[43] = 0      (target of STOREI; will become 100)
      MEM[44] = 25     (operand used by AND/OR/SUB)
      MEM[45] = 0      (will be written by STORE with 100)
      MEM[60] = 100    (operand for SUB to force negative -> N=1)
      MEM[61] = 1      (operand used on the fallthrough path if JN not taken)
    """
    # check for -v flag for verbose debugging prints
    DEBUG = True if sys.argv.count('-v') > 0 else False
    debug = False

    # process file passed through command line
    arg_mem_file = None
    if len(sys.argv) > 1 and "mem" in sys.argv[1]:
        arg_mem_file = sys.argv[1]
    if arg_mem_file:
        initial_mem = process_mem_file(arg_mem_file)
    else:
        DEBUG = True
        debug = True
        program = [
            #  0
            Instruction('LOADI', 40),      # ACC <- M[M[40]] = M[41] = 50
            Instruction('ADD', 41),       # 3  ACC <- ACC + M[41] = 50 + 50 = 100
            Instruction('STOREI', 42),    # 6  M[M[42]] <- ACC  => M[43] = 100
            Instruction('STORE', 45),     # 9  M[45] <- ACC  => 100
            Instruction('LOAD', 45),      # 11 ACC <- M[45] = 100
            Instruction('AND', 44),       # 14 ACC <- 100 & 25 = 0
            Instruction('OR', 44),        # 17 ACC <- 0 | 25 = 25
            Instruction('SUB', 44),       # 20 ACC <- 25 - 25 = 0  (Z=1, N=0)
            Instruction('SUB', 60),       # 23 ACC <- 0 - 100 = 156 (0x9C), N=1
            # SHRINK 25 to adjust for removed NOPs
            Instruction('JN', 13),        # 26 if N=1 jump to index 25 (target below)
            # Fallthrough path (should be skipped because N=1)
            Instruction('ADD', 61),       # 29 would do ACC <- ACC + 1
            Instruction('STORE', 50),     # 30 would store to M[50]
            Instruction('HALT'),          # 31 would halt if we didn't jump
            # Jump target:
            Instruction('LOAD', 43),      # 32 ACC <- M[30] = 100  (stored earlier by STOREI)
            Instruction('HALT'),          # 34 final stop
        ]

        preloaded_mem = {
            40: 41,
            41: 50,
            42: 43,
            43: 0,
            44: 25,
            45: 0,
            60: 100,
            61: 1,
        }

        initial_mem = {}
        for i, instr in enumerate(program):
            initial_mem[i] = instr.instruction_to_binary()
        for k, v in preloaded_mem.items():
            initial_mem[k] = v & 0xFF

    # Initial architectural state
    print("Initial Memory: ")
    for loc in sorted(initial_mem.keys()):
        val = initial_mem[loc]
        Instruction_inst = Instruction.binary_to_instruction(val)
        print(f"MEM[{loc:02}] = {Instruction_inst.opcode_to_string()} {Instruction_inst.operand}")

    myCPU = S12PipelineCPU(initial_mem)
    # set pipelined to false for benchmarking non-pipelined execution
    # -p flag can be used to enable pipelined execution
    pipelined = False if sys.argv.count('-n') > 0 else True
    # Check for branch method using -b switch
    myCPU.branch_prediction["method"] = "none"   #Default: Branch Prediction Disabled
    if "-b" in sys.argv:
        index = sys.argv.index("-b")
        if index + 1 < len(sys.argv):   # make sure there is something after -b
            myCPU.branch_prediction["method"] = sys.argv[index + 1]

    total_cycles = myCPU.run(max_cycles=20000, start_pc=0, pipelined=pipelined)

    if debug:
        # Final architectural state
        print("\nFINAL STATE")
        print("ACC:", myCPU.ACC, "Z:", myCPU.Z, "N:", myCPU.N)
        print(f"MEM[41]: {myCPU.MEM[41]}, MEM[42]: {myCPU.MEM[42]}, MEM[43]: {myCPU.MEM[43]}, MEM[44]: {myCPU.MEM[44]}, MEM[45]: {myCPU.MEM[45]}")
        print(f"Total cycles executed: {total_cycles}")

        # Simple checks 
        assert myCPU.ACC == 100 and myCPU.Z == 0 and myCPU.N == 0, "Final ACC/Z/N mismatch"
        assert myCPU.MEM[43] == 100, "MEM[43] should be 100 (written by STOREI)"
        assert myCPU.MEM[45] == 100, "MEM[40] should be 100 (written by STORE)"
        assert myCPU.MEM.get(50, 0) == 0, "MEM[50] should be 0 (fallthrough skipped by JN)"
        print(" All expected results match.")

    # write memory to -o file if specified
    output_mem_file = None
    if '-o' in sys.argv:
        o_index = sys.argv.index('-o')
        if o_index + 1 < len(sys.argv):
            output_mem_file = sys.argv[o_index + 1]
    if output_mem_file:
        write_mem_file(output_mem_file, myCPU.MEM)
    myCPU.display_performance_counters()
    myCPU.display_branch_prediction_metrics()
