# From: -> src/runner.py
# MixLang Project -> 16/8/2025 
# Made by tienanh109 (Le Tien Anh) with GPL-3 Licence
# https://github.com/tienanh109/TAZ-MixLang/

import argparse
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

# --- PROJECT METADATA ---
__VERSION__ = "1.1"
__AUTHOR__ = "tienanh109"
__DOCS_URL__ = "https://tienanh109.github.io/TAZ-MixLang/" 

# --- PART 1: MODELS ---

@dataclass
class CodeBlock:
    """Represents a block of code from the .mix file."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extension: str = ""
    interpreter: str = ""
    code: str = ""
    task_id: str = ""
    line_start: int = 0
    temp_file_path: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

@dataclass
class Task:
    """Represents a named task containing one or more CodeBlocks."""
    id: str
    blocks: List[CodeBlock] = field(default_factory=list)

# --- PART 2: LOGGER ---

class CustomFormatter(logging.Formatter):
    """Custom formatter to add prefixes, handling Windows' lack of color support."""
    def format(self, record):
        interpreter = getattr(record, 'interpreter', 'SYSTEM')
        if os.name == 'nt':
             return f"[{interpreter.upper()}] {record.getMessage()}"
        
        color_map = {"python": "\033[94m", "nodejs": "\03d[92m", "bash": "\033[93m", "cmd": "\033[95m"}
        color = color_map.get(interpreter, "\033[0m")
        end_color = "\033[0m"
        return f"{color}[{interpreter.upper()}]{end_color} {record.getMessage()}"

def setup_logger():
    logger = logging.getLogger("MixRunner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    return logger

log = setup_logger()

# --- PART 3: CORE LOGIC ---

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

class MixParser:
    """Parses a .mix file into a list of executable Tasks and handles global commands."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header_pattern = re.compile(
            r'call\("(?P<ext>\.\w+)"\)\.from\((?P<interp>\w+)\)(?:\.task\("(?P<task_id>\w+)"\))?:')
        self.global_commands = []

    def _dedent_code(self, code: str) -> str:
        """Removes common leading whitespace from a code block."""
        lines = code.splitlines()
        
        # Find the first non-empty line to determine the base indentation
        first_line_index = -1
        for i, line in enumerate(lines):
            if line.strip():
                first_line_index = i
                break
        if first_line_index == -1: return ""

        # Calculate the indentation of the first content line
        first_line = lines[first_line_index]
        indent_len = len(first_line) - len(first_line.lstrip())
        indent = first_line[:indent_len]

        # Dedent all lines
        dedented_lines = []
        for line in lines:
            if line.startswith(indent):
                dedented_lines.append(line[indent_len:])
            else:
                dedented_lines.append(line.strip())
        return '\n'.join(dedented_lines)

    def parse(self) -> List[Task]:
        tasks: Dict[str, Task] = {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            log.error(f"Error: File not found at '{self.file_path}'")
            return []

        in_block = False
        current_block = None
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped_line = line.strip()

            if stripped_line.startswith('#') or not stripped_line:
                continue

            if not in_block:
                if stripped_line == 'clear();':
                    self.global_commands.append(clear_screen)
                    continue

                match = self.header_pattern.match(stripped_line)
                if match:
                    data = match.groupdict()
                    task_id = data.get('task_id') or f"independent_{uuid.uuid4().hex[:8]}"
                    if task_id not in tasks: tasks[task_id] = Task(id=task_id)
                    current_block = CodeBlock(
                        extension=data['ext'], interpreter=data['interp'],
                        task_id=task_id, line_start=line_num
                    )
                    in_block = True
                elif stripped_line:
                    log.error(f"[Error:{line_num}:1]: Invalid syntax. Expected 'call(...)' definition or global command.")
                    return []
            else:
                if stripped_line == 'call(done)':
                    if current_block:
                        # [FIX] Dedent the collected code before adding it
                        current_block.code = self._dedent_code(current_block.code)
                        tasks[current_block.task_id].blocks.append(current_block)
                    in_block = False
                    current_block = None
                else:
                    if current_block:
                        # Note: We now add the raw line, dedenting happens at the end
                        current_block.code += line
        
        if in_block:
            start = current_block.line_start if current_block else 'N/A'
            log.error(f"[Error:{start}:1]: Unterminated block. Missing 'call(done)'.")
            return []

        return list(tasks.values())

class TempFileGenerator:
    """Creates temporary script files for execution."""
    def __init__(self, run_id: str):
        self.base_dir = Path(f"temp/{run_id}")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_temp_file(self, block: CodeBlock):
        task_dir = self.base_dir / block.task_id
        task_dir.mkdir(exist_ok=True)
        file_path = task_dir / f"{block.id}{block.extension}"
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(block.code)
        os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IEXEC)
        block.temp_file_path = str(file_path)

def execute_block(block: CodeBlock, hide_log: bool = False) -> CodeBlock:
    """Executes a single CodeBlock."""
    if not block.temp_file_path:
        raise ValueError("Temp file path not set.")

    logger = logging.LoggerAdapter(log, {'interpreter': block.interpreter})
    if not hide_log:
        logger.info(f"Executing block {block.id[:8]} from task '{block.task_id}'...")
    
    interpreter_path = shutil.which(block.interpreter)
    if block.interpreter == 'python' and not interpreter_path:
        interpreter_path = sys.executable

    if not interpreter_path:
        block.stderr = f"Error: Interpreter '{block.interpreter}' not found in PATH."
        block.exit_code = 127
        logger.error(block.stderr)
        return block

    command = [interpreter_path, block.temp_file_path]
    if block.interpreter == 'cmd':
        command = [interpreter_path, '/c', block.temp_file_path]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='backslashreplace'
        )
        for line in process.stdout:
            block.stdout += line
            if hide_log: sys.stdout.write(line)
            else: logger.info(line.strip())
        
        stderr_output = process.communicate()[1]
        block.stderr = stderr_output
        if stderr_output:
            logger.error(f"[Error:{block.line_start}:1]: {stderr_output.strip()}")

        block.exit_code = process.returncode
        if not hide_log:
            status = "succeeded" if block.exit_code == 0 else f"failed (code: {block.exit_code})"
            logger.info(f"Block {block.id[:8]} finished, status: {status}.")
    except Exception as e:
        block.stderr = f"An unexpected error occurred during execution: {e}"
        block.exit_code = 1
        logger.error(f"[Error:{block.line_start}:1]: {block.stderr}")
    return block

class Scheduler:
    """Schedules and runs tasks concurrently."""
    def __init__(self, tasks: List[Task], max_parallel: int, hide_log: bool = False):
        self.tasks = tasks
        self.max_parallel = max_parallel
        self.hide_log = hide_log

    def run(self) -> bool:
        start_time = time.time()
        all_blocks = [block for task in self.tasks for block in task.blocks]
        total_blocks = len(all_blocks)
        failed_blocks = 0

        if not self.hide_log:
            log.info(f"Starting run of {len(self.tasks)} tasks ({total_blocks} blocks) with max {self.max_parallel} parallel jobs.")

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_block = {executor.submit(execute_block, block, self.hide_log): block for block in all_blocks}
            for i, future in enumerate(as_completed(future_to_block), 1):
                result_block = future.result()
                if result_block.exit_code != 0:
                    failed_blocks += 1
                if not self.hide_log:
                    log.info(f"Progress: {i}/{total_blocks} blocks completed.")

        if not self.hide_log:
            duration = time.time() - start_time
            log.info("-" * 50)
            log.info("RUN SUMMARY")
            log.info(f"Total execution time: {duration:.2f} seconds")
            log.info(f"Succeeded: {total_blocks - failed_blocks}/{total_blocks}")
            log.info(f"Failed: {failed_blocks}")
            log.info("-" * 50)
        return failed_blocks == 0

def cleanup_temp_files(run_id: str, hide_log: bool = False):
    temp_dir = Path(f"temp/{run_id}")
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            if not hide_log: log.info(f"Cleaned up temporary directory: {temp_dir}")
        except OSError as e:
            if not hide_log: log.error(f"Error cleaning up temp directory {temp_dir}: {e}")

# --- PART 4: MAIN ENTRYPOINT ---

def main():
    """Main function to parse arguments and run the interpreter."""
    help_epilog = f"""
SYNTAX:
  call(".ext").from(interpreter).task("task_name"):
      # Your code here (can be indented)
  call(done)

  - Blocks with the same task("task_name") run in parallel.
  - Blocks without a .task(...) run independently.
  - Use 'clear();' on its own line to clear the console screen.

EXAMPLE:
  # build_and_test.mix
  call(".py").from(python).task("build"):
      print("Building Python part...")
  call(done)

  clear();

  call(".js").from(nodejs).task("test"):
      console.log("Running Node.js tests...");
  call(done)

taz-mixlang v{__VERSION__} by {__AUTHOR__}
"""
    parser = argparse.ArgumentParser(
        description="A polyglot script runner for .mix files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=help_epilog
    )
    parser.add_argument("file", nargs='?', default=None, help="Path to the .mix file to execute.")
    parser.add_argument("--max-parallel", type=int, default=4, help="Maximum number of blocks to run in parallel.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files after execution for debugging.")
    parser.add_argument("--hide-log", action="store_true", help="Only show script stdout, hiding all system logs.")
    parser.add_argument("--info", action="store_true", help="Show project information and exit.")
    
    args = parser.parse_args()

    if args.info:
        print(f"taz-mixlang (Mix Runner)")
        print(f"Version: {__VERSION__}")
        print(f"Author: {__AUTHOR__}")
        print(f"Docs: {__DOCS_URL__}")
        sys.exit(0)

    if not args.file:
        parser.print_help()
        sys.exit(1)

    run_id = uuid.uuid4().hex
    if not args.hide_log: log.info(f"Run ID: {run_id}")

    parser_instance = MixParser(args.file)
    tasks = parser_instance.parse()
    
    for command in parser_instance.global_commands:
        command()

    if not tasks and not parser_instance.global_commands:
        sys.exit(1)
    
    generator = TempFileGenerator(run_id)
    for task in tasks:
        for block in task.blocks:
            generator.create_temp_file(block)
    
    scheduler = Scheduler(tasks, args.max_parallel, hide_log=args.hide_log)
    success = scheduler.run()

    if not args.keep_temp:
        cleanup_temp_files(run_id, hide_log=args.hide_log)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
