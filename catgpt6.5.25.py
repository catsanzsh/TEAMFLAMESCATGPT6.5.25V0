# test.py

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tkinter as tk
import importlib.util
import requests
import traceback
import queue
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from types import ModuleType
from typing import Any, Dict, List, Tuple, Optional, Union, Callable, Awaitable

try:
    import aiohttp
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False

from tkinter import messagebox, scrolledtext, simpledialog

# ────────────────────────────── Logging Setup ───────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# ────────────────────────────── Runtime Globals ─────────────────────────────
RUNTIME_API_KEY: Optional[str] = None # Will store API key in memory

# ────────────────────────────── Runtime Paths ───────────────────────────────
HOME = Path.home()
ARCHIVE_DIR = HOME / "Documents" / "CatGPT_Agent_Archive"
PLUGIN_DIR = ARCHIVE_DIR / "plugins"
AGENT_WORKSPACE_DIR = ARCHIVE_DIR / "autonomous_workspace"
MEMORY_FILE = ARCHIVE_DIR / "memory.json"
MODEL_FILE = ARCHIVE_DIR / "models.json"
ARCHIVE_INDEX = ARCHIVE_DIR / "archive_index.json"

# Ensure directories exist
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
AGENT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── Constants ──────────────────────────────────
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY" # Environment variable name
DEFAULT_MODELS = ["meta-llama/llama-3-8b-instruct", "gpt-3.5-turbo", "claude-3-opus", "gpt-4"]
LLM_TIMEOUT = 120  # seconds
CODE_EXEC_TIMEOUT = 60 # Timeout for sandboxed code execution

# UI Theme elements
UI_THEME = {
    "bg_primary": "#f5f5f5", "bg_secondary": "#ffffff", "bg_tertiary": "#f0e6ff",
    "bg_chat_display": "#fafafa", "bg_chat_input": "#f9f9f9", "bg_editor": "#1e2838",
    "bg_editor_header": "#34495e", "bg_button_primary": "#10a37f", "bg_button_secondary": "#3498db",
    "bg_button_danger": "#e74c3c", "bg_button_warning": "#f39c12", "bg_button_evolution": "#9b59b6",
    "bg_button_evo_compile": "#27ae60", "bg_button_info": "#5dade2", "bg_listbox_select": "#6c5ce7",
    "bg_mission_control": "#2c3e50", "fg_mission_control": "#ecf0f1",
    "fg_primary": "#2c3e50", "fg_secondary": "#ecf0f1", "fg_button_light": "#ffffff",
    "fg_evolution_header": "#6c5ce7", "font_default": ("Consolas", 11), "font_chat": ("Consolas", 11),
    "font_button_main": ("Arial", 11, "bold"), "font_button_small": ("Arial", 10),
    "font_title": ("Arial", 14, "bold"), "font_editor": ("Consolas", 11), "font_listbox": ("Consolas", 9),
    "font_mission_control": ("Consolas", 10)
}

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def now_ts() -> str:
    """Generates a high-resolution timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def get_api_key() -> str:
    """Returns the API key stored in memory."""
    if RUNTIME_API_KEY is not None:
        return RUNTIME_API_KEY
    logger.warning("API Key not found in memory. Ensure it was set at startup.")
    return ""

def get_current_source_code() -> str:
    """Retrieves the source code of the currently running script."""
    try:
        return inspect.getsource(sys.modules[__name__])
    except:
        return "# Error: Could not retrieve current source code."

# ----------------------------------------------------------------------------
# Optimized API Client
# ----------------------------------------------------------------------------
class APIClient:
    """Handles all communication with the LLM API, managing sessions efficiently."""
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key_getter: Callable[[], str], timeout: int):
        self._api_key_getter = api_key_getter
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed: self._session = aiohttp.ClientSession()
        return self._session

    def _get_headers(self) -> Dict[str, str]:
        api_key = self._api_key_getter()
        if not api_key: raise RuntimeError("API Key is missing. Please configure it.")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _parse_response(self, data: Dict[str, Any]) -> str:
        try: return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected LLM API response structure: {data}. Error: {e}")
            raise RuntimeError("Invalid response structure from LLM API.")

    async def call_async(self, payload: Dict[str, Any]) -> str:
        session = await self._get_session()
        try:
            async with session.post(self.API_URL, headers=self._get_headers(), json=payload, timeout=self.timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"LLM API call failed with status {resp.status}: {error_text}")
                    raise RuntimeError(f"API Error (Status {resp.status}): {error_text}")
                return self._parse_response(await resp.json())
        except aiohttp.ClientError as e: raise RuntimeError(f"Network Error: {e}")

    def call_sync(self, payload: Dict[str, Any]) -> str:
        try:
            response = requests.post(self.API_URL, headers=self._get_headers(), json=payload, timeout=self.timeout)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.RequestException as e: raise RuntimeError(f"Network Error: {e}")

    async def close_session(self):
        if self._session and not self._session.closed: await self._session.close()

# ----------------------------------------------------------------------------
# Code Interpreter for Sandboxed Execution
# ----------------------------------------------------------------------------
class CodeInterpreter:
    def __init__(self, timeout: int = CODE_EXEC_TIMEOUT, workspace_dir: Path = AGENT_WORKSPACE_DIR):
        self.timeout = timeout
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Code interpreter will use workspace: {self.workspace_dir}")

    def execute_code(self, code_string: str) -> Tuple[str, str, Optional[str]]:
        stdout_str, stderr_str, error_msg = "", "", None
        temp_script_path = self.workspace_dir / f"temp_script_{now_ts()}.py"
        try:
            temp_script_path.write_text(code_string, encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "-u", str(temp_script_path)],
                capture_output=True, text=True, timeout=self.timeout,
                cwd=str(self.workspace_dir), check=False
            )
            stdout_str, stderr_str = process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            error_msg = f"Code execution timed out after {self.timeout} seconds."
            stderr_str += f"\nTimeoutError: Execution exceeded {self.timeout} seconds."
        except Exception as e:
            error_msg = f"An unexpected error occurred during subprocess execution: {e}"
        finally:
            if temp_script_path.exists(): temp_script_path.unlink()
        return stdout_str, stderr_str, error_msg

# ----------------------------------------------------------------------------
# AutonomousAgent (AutoGPT Core Logic)
# ----------------------------------------------------------------------------
class AutonomousAgent:
    """A goal-driven agent that runs a continuous think-act loop."""
    def __init__(self, goal: str, api_client: APIClient, code_interpreter: CodeInterpreter,
                 model_cfg: Dict, ui_queue: queue.Queue, stop_event: Event):
        self.goal = goal
        self.api_client = api_client
        self.code_interpreter = code_interpreter
        self.model_cfg = model_cfg
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.history: List[Dict[str, str]] = []
        self.tools = {
            "execute_python_code": self.code_interpreter.execute_code,
            "write_file": self.write_file,
            "read_file": self.read_file,
            "list_files": self.list_files,
            "task_complete": self.task_complete
        }

    def log_to_ui(self, message: str, tag: str = "info"):
        """Safely puts messages onto the UI queue."""
        self.ui_queue.put({"tag": tag, "content": message})

    def get_system_prompt(self) -> str:
        """Generates the master prompt that guides the autonomous agent."""
        tool_definitions = [
            {"name": "execute_python_code", "description": "Executes a string of Python code in a sandboxed environment within the agent's workspace. Returns stdout, stderr, and any system errors.", "args": {"code_string": "The Python code to execute as a single string."}},
            {"name": "write_file", "description": "Writes content to a specified file in the agent's workspace. Overwrites if the file exists.", "args": {"filename": "The name of the file.", "content": "The content to write."}},
            {"name": "read_file", "description": "Reads the entire content of a specified file from the agent's workspace.", "args": {"filename": "The name of the file to read."}},
            {"name": "list_files", "description": "Lists all files and directories in the agent's workspace.", "args": {}},
            {"name": "task_complete", "description": "Call this function ONLY when you have fully completed the user's goal. This will shut down the agent.", "args": {"reason": "A summary of why you believe the task is complete."}}
        ]
        return (
            "You are an autonomous AI agent, CatGPT-AGI. Your decisions must be made independently without user assistance. "
            "Your primary goal is: {goal}\n\n"
            "You operate in a continuous loop of Thought -> Plan -> Command -> Result.\n"
            "1. **Thought:** First, reflect on your goal and the last result. Reason about the best next step.\n"
            "2. **Plan:** Formulate a short, numbered list of steps you will take in your next action.\n"
            "3. **Command:** Issue a single command from the available tools. You MUST format your response as a single, valid JSON object with 'thought', 'plan', and 'command' keys. The 'command' must be an object with 'name' and 'args' keys.\n\n"
            "Available tools:\n{tools}\n\n"
            "RESPONSE FORMAT:\n"
            "```json\n"
            "{{\n"
            '  "thought": "Your reasoning for the next action.",\n'
            '  "plan": [\n'
            '    "Step 1: First part of the action.",\n'
            '    "Step 2: Second part of the action."\n'
            '  ],\n'
            '  "command": {{\n'
            '    "name": "tool_name",\n'
            '    "args": {{"arg1": "value1"}}\n'
            '  }}\n'
            "}}\n"
            "```\n"
            "Your first task is to think about how to achieve your goal: '{goal}'. Start by outlining a high-level plan."
        ).format(goal=self.goal, tools=json.dumps(tool_definitions, indent=2))

    # --- Tool Implementations ---
    def write_file(self, filename: str, content: str) -> str:
        try:
            (AGENT_WORKSPACE_DIR / filename).write_text(content, encoding='utf-8')
            return f"Successfully wrote to '{filename}'."
        except Exception as e: return f"Error writing to file: {e}"

    def read_file(self, filename: str) -> str:
        try: return (AGENT_WORKSPACE_DIR / filename).read_text(encoding='utf-8')
        except FileNotFoundError: return f"Error: File '{filename}' not found."
        except Exception as e: return f"Error reading file: {e}"

    def list_files(self) -> str:
        try:
            files = [str(p.relative_to(AGENT_WORKSPACE_DIR)) for p in AGENT_WORKSPACE_DIR.rglob("*")]
            return "Workspace files:\n" + "\n".join(files) if files else "Workspace is empty."
        except Exception as e: return f"Error listing files: {e}"

    def task_complete(self, reason: str) -> str:
        self.stop_event.set()
        return f"TASK COMPLETE. Reason: {reason}"

    def run(self):
        """The main execution loop for the agent."""
        self.log_to_ui(f"AUTONOMOUS AGENT ACTIVATED\nGOAL: {self.goal}\n", "header")
        
        self.history.append({"role": "system", "content": self.get_system_prompt()})

        while not self.stop_event.is_set():
            try:
                # 1. THINK
                self.log_to_ui("Thinking...", "status")
                payload = self.model_cfg.copy()
                payload['messages'] = self.history[-10:] # Keep context manageable
                
                llm_response_raw = self.api_client.call_sync(payload)

                # 2. PARSE COMMAND
                self.log_to_ui(f"Received raw response from LLM:\n{llm_response_raw}", "llm")
                match = re.search(r"```json\s*\n(.*?)\n```", llm_response_raw, re.DOTALL)
                if not match:
                    self.log_to_ui("ERROR: LLM did not return a valid JSON code block. Retrying...", "error")
                    self.history.append({"role": "user", "content": "Your response was not in the required JSON format. Please try again."})
                    continue

                parsed_json = json.loads(match.group(1).strip())
                thought = parsed_json.get("thought", "No thought provided.")
                plan = "\n".join(f"- {p}" for p in parsed_json.get("plan", []))
                command = parsed_json.get("command", {})
                command_name = command.get("name")
                command_args = command.get("args", {})

                self.log_to_ui(f"THOUGHT: {thought}\nPLAN:\n{plan}\nCOMMAND: {command_name}({command_args})", "thought")

                # 3. EXECUTE
                if command_name in self.tools:
                    tool_func = self.tools[command_name]
                    # Special handling for code execution tuple result
                    if command_name == 'execute_python_code':
                        stdout, stderr, exec_err = tool_func(**command_args)
                        result = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                        if exec_err: result += f"\nEXECUTION_ERROR: {exec_err}"
                    else:
                        result = tool_func(**command_args)
                else:
                    result = f"Error: Unknown command '{command_name}'."
                
                self.log_to_ui(f"RESULT:\n{result}", "result")

                # 4. UPDATE HISTORY
                self.history.append({"role": "assistant", "content": llm_response_raw})
                self.history.append({"role": "user", "content": f"Command `{command_name}` executed. Result:\n{result}"})

            except json.JSONDecodeError as e:
                self.log_to_ui(f"ERROR: Failed to decode JSON from LLM response: {e}", "error")
                self.history.append({"role": "user", "content": "Your last response was not valid JSON. Please correct it."})
            except Exception as e:
                tb = traceback.format_exc()
                self.log_to_ui(f"CRITICAL AGENT ERROR: {e}\n{tb}", "error")
                self.stop_event.set()

        self.log_to_ui("Autonomous agent has shut down.", "header")

# ----------------------------------------------------------------------------
# DarwinAgent (Interactive Chat Agent)
# ----------------------------------------------------------------------------
class DarwinAgent:
    def __init__(self, ui_app_ref):
        self.ui_app = ui_app_ref
        self.models: List[str] = self._load_models()
        self.cfg: Dict[str, Any] = {"model": self.models[0], "temperature": 0.7, "max_tokens": 4096}
        self.history: List[Dict[str, str]] = self._load_memory()
        self.agent_archive: List[Tuple] = self._load_agent_archive()
        self.api_client = APIClient(get_api_key, LLM_TIMEOUT)
        self.code_interpreter = CodeInterpreter(timeout=CODE_EXEC_TIMEOUT)
        logger.info("DarwinAgent initialized.")

    def _load_json_file(self, fp, default): return json.loads(fp.read_text()) if fp.exists() else default
    def _save_json_file(self, fp, data): fp.write_text(json.dumps(data, indent=2))
    def _load_memory(self): return self._load_json_file(MEMORY_FILE, [])
    def _save_memory(self): self._save_json_file(MEMORY_FILE, self.history[-2000:])
    def _load_models(self):
        models = self._load_json_file(MODEL_FILE, [])
        if not models: self._save_json_file(MODEL_FILE, DEFAULT_MODELS); return DEFAULT_MODELS
        return models
    def _load_agent_archive(self): return self._load_json_file(ARCHIVE_INDEX, [])
    def _save_agent_archive(self): self._save_json_file(ARCHIVE_INDEX, self.agent_archive)

    def _prepare_payload(self) -> Dict[str, Any]:
        system_prompt = ("You are DarwinCat, a helpful AI assistant. To execute Python code, respond with ```python_exec...```.")
        messages = list(self.history)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        return {"model": self.cfg["model"], "messages": messages[-20:], "temperature": self.cfg["temperature"], "max_tokens": self.cfg["max_tokens"]}

    def _extract_executable_code(self, msg: str) -> Optional[str]:
        return (m.group(1).strip() if (m := re.search(r"```python_exec\s*\n(.*?)\n```", msg, re.DOTALL)) else None)

    def _format_code_execution_summary(self, stdout: str, stderr: str, exec_error: Optional[str]) -> str:
        summary = "--- Python Code Execution ---\n"
        if stdout: summary += f"STDOUT:\n```text\n{stdout.strip()}\n```\n"
        if stderr: summary += f"STDERR:\n```text\n{stderr.strip()}\n```\n"
        if exec_error: summary += f"EXECUTION SYSTEM ERROR: {exec_error}\n"
        if not (stdout or stderr or exec_error): summary += "Code executed successfully with no output.\n"
        return summary.strip()

    async def _ask_orchestrator(self, user_msg: str) -> str:
        if user_msg.startswith("/model "):
            mdl = user_msg.split(maxsplit=1)[1]
            self.cfg["model"] = mdl
            return f"Model switched to {mdl}"

        if user_msg.startswith("/imagine "):
            goal = user_msg.split(maxsplit=1)[1]
            self.ui_app.after(0, self.ui_app.launch_autonomous_agent_window, goal)
            return f"AUTONOMOUS AGENT DISPATCHED!\nGoal: '{goal}'\nA new 'Mission Control' window has opened."
        
        self.history.append({"role": "user", "content": user_msg})
        try:
            payload = self._prepare_payload()
            assistant_msg_1 = await self.api_client.call_async(payload)
            self.history.append({"role": "assistant", "content": assistant_msg_1})

            if code_to_execute := self._extract_executable_code(assistant_msg_1):
                stdout, stderr, exec_err = await asyncio.to_thread(self.code_interpreter.execute_code, code_to_execute)
                summary = self._format_code_execution_summary(stdout, stderr, exec_err)
                feedback = f"The Python code was executed. Result:\n{summary}"
                self.history.append({"role": "user", "content": feedback})
                payload_2 = self._prepare_payload()
                assistant_msg_2 = await self.api_client.call_async(payload_2)
                self.history.append({"role": "assistant", "content": assistant_msg_2})
                self._save_memory()
                return f"{assistant_msg_1}\n\n{summary}\n\nCatGPT (after execution):\n{assistant_msg_2}"
            
            self._save_memory()
            return assistant_msg_1
        except RuntimeError as e: return f"[LLM-Error] {e}"
        except Exception as e: return f"[Agent-Error] An unexpected error occurred: {e}"

    async def ask_async(self, user_msg: str) -> str: return await self._ask_orchestrator(user_msg)
    def ask_sync(self, user_msg: str) -> str: return asyncio.run(self._ask_orchestrator(user_msg))

    def recompile(self, new_code: str) -> Tuple[str, str, Optional[str]]:
        ts = now_ts(); fname_stem = f"CatGPT_Agent_v{ts}"; agent_file_path = ARCHIVE_DIR / f"{fname_stem}.py"
        # Dummy compile check for brevity
        status, error = ("FIT", None) if "class DarwinAgent" in new_code else ("QUARANTINED", "Missing DarwinAgent class")
        final_code = f'"""\nTimestamp: {datetime.now()}\nStatus: {status}\n"""\n\n' + new_code
        agent_file_path.write_text(final_code, encoding="utf-8")
        self.agent_archive.append((ts, agent_file_path.name, status, str(error)))
        self._save_agent_archive()
        return agent_file_path.name, status, str(error)

    async def shutdown(self): await self.api_client.close_session()

# ----------------------------------------------------------------------------
# Tkinter UI Layer
# ----------------------------------------------------------------------------
class CatGPTFusion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Darwin CatGPT Fusion Edition")
        self.geometry("1100x750")
        self.config(bg=UI_THEME["bg_primary"])
        self._prompt_for_api_key_if_missing()
        self.agent = DarwinAgent(self)
        self.intro_message = "Welcome to Darwin CatGPT! Use /imagine <your goal> to launch the autonomous agent."
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        logger.info("CatGPTFusion UI initialized.")

    def _prompt_for_api_key_if_missing(self):
        global RUNTIME_API_KEY
        if env_key := os.environ.get(OPENROUTER_API_KEY_ENV_VAR): RUNTIME_API_KEY = env_key.strip()
        else: RUNTIME_API_KEY = simpledialog.askstring("API Key Required", "Enter your OpenRouter API Key:") or ""
        if not RUNTIME_API_KEY: messagebox.showwarning("API Key Missing", "No API Key entered. AI features may fail.")

    def _build_ui(self):
        main = tk.Frame(self, bg=UI_THEME["bg_primary"]); main.pack(fill="both", expand=True, padx=10, pady=10)
        left = tk.Frame(main, bg=UI_THEME["bg_secondary"], relief=tk.RAISED, bd=1); left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = tk.Frame(main, bg=UI_THEME["bg_tertiary"], relief=tk.RIDGE, bd=2); right.pack(side="right", fill="y", padx=(5, 0))
        self._build_chat_display(left); self._build_input_area(left); self._build_control_buttons(left)
        self._build_archive_panel(right); self._display_initial_messages()

    def _build_chat_display(self, p):
        self.chat_window = scrolledtext.ScrolledText(p, bg=UI_THEME["bg_chat_display"], font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1)
        self.chat_window.pack(fill="both", expand=True, padx=10, pady=10); self.chat_window.configure(state=tk.DISABLED)

    def _build_input_area(self, p):
        f = tk.Frame(p, bg=UI_THEME["bg_secondary"]); f.pack(fill="x", padx=10, pady=(0, 10))
        self.input_field = tk.Text(f, height=3, bg=UI_THEME["bg_chat_input"], font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1)
        self.input_field.pack(side="left", fill="x", expand=True); self.input_field.bind("<Return>", lambda e: self._on_send() if not (e.state & 0x1) else None)
        tk.Button(f, text="Send", command=self._on_send, bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_main"]).pack(side="right", padx=5, fill="y")

    def _build_control_buttons(self, p):
        f = tk.Frame(p, bg=UI_THEME["bg_secondary"]); f.pack(fill="x", padx=10, pady=(0, 10))
        btns = [("Recompile", self._agent_recompile_window), ("Clear History", self._clear_history)]
        for txt, cmd in btns: tk.Button(f, text=txt, command=cmd, bg=UI_THEME["bg_button_secondary"], fg=UI_THEME["fg_button_light"]).pack(side="left", padx=2)

    def _build_archive_panel(self, p):
        tk.Label(p, text="🧬 Evolution Archive", font=UI_THEME["font_title"], bg=UI_THEME["bg_tertiary"], fg=UI_THEME["fg_evolution_header"]).pack(pady=8)
        self.archive_listbox = tk.Listbox(p, width=50, font=UI_THEME["font_listbox"], relief=tk.SOLID, bd=1)
        self.archive_listbox.pack(fill="both", expand=True, padx=5)
        self._refresh_archive_listbox()

    def _display_initial_messages(self):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"CatGPT: {self.intro_message}\n")
        self.chat_window.insert(tk.END, f"Models: {', '.join(self.agent.models)}\n")
        self.chat_window.insert(tk.END, "Commands: /model <name>, /imagine <goal>\n")
        self.chat_window.config(state=tk.DISABLED)

    def _append_chat(self, who: str, txt: str):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"\n{who}:\n{txt}\n"); self.chat_window.see(tk.END)
        self.chat_window.config(state=tk.DISABLED)

    def _on_send(self):
        user_msg = self.input_field.get("1.0", "end-1c").strip()
        if not user_msg: return
        self.input_field.delete("1.0", tk.END); self._append_chat("You", user_msg)
        Thread(target=self._worker, args=(user_msg,), daemon=True).start()

    def _worker(self, msg: str):
        try:
            answer = asyncio.run(self.agent.ask_async(msg)) if ASYNC_MODE else self.agent.ask_sync(msg)
        except Exception as e: answer = f"[error] An unexpected error occurred: {e}"
        if self.winfo_exists(): self.after(0, lambda: self._append_chat("CatGPT", answer))

    def _clear_history(self):
        if messagebox.askyesno("Confirm", "Clear chat history and memory?"):
            self.agent.history.clear(); self.agent._save_memory(); self.chat_window.config(state=tk.NORMAL)
            self.chat_window.delete('1.0', tk.END); self._display_initial_messages()

    def _on_closing(self):
        if ASYNC_MODE and hasattr(self, 'agent'): asyncio.run(self.agent.shutdown())
        self.destroy()

    def _agent_recompile_window(self):
        win = tk.Toplevel(self); win.title("Agent Recompile"); win.geometry("900x700"); win.transient(self); win.grab_set()
        editor = tk.Text(win, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"], font=UI_THEME["font_editor"], undo=True)
        editor.pack(fill="both", expand=True, padx=10, pady=10); editor.insert(tk.END, get_current_source_code())
        def compile_and_evolve():
            _, status, _ = self.agent.recompile(editor.get("1.0", "end-1c"))
            self._refresh_archive_listbox(); (messagebox.showinfo if status=="FIT" else messagebox.showwarning)("Evolution Result", f"Status: {status}", parent=win); win.destroy()
        tk.Button(win, text="🚀 Compile & Evolve", command=compile_and_evolve, bg=UI_THEME["bg_button_evo_compile"], fg=UI_THEME["fg_button_light"]).pack(pady=10)

    def _refresh_archive_listbox(self):
        self.archive_listbox.delete(0, tk.END)
        for ts, filename, status, _ in reversed(self.agent.agent_archive):
            icon = "✅" if status == "FIT" else "🔒"
            self.archive_listbox.insert(tk.END, f"{icon} {ts.split('_')[0]} | {filename}")

    # --- AUTONOMOUS AGENT UI ---
    def launch_autonomous_agent_window(self, goal: str):
        win = tk.Toplevel(self); win.title("CatGPT Mission Control"); win.geometry("600x400")
        win.config(bg=UI_THEME["bg_mission_control"])

        log_display = scrolledtext.ScrolledText(win, wrap=tk.WORD, bg="#1C1C1C", fg=UI_THEME["fg_mission_control"], font=UI_THEME["font_mission_control"])
        log_display.pack(fill="both", expand=True, padx=5, pady=5)
        log_display.tag_config("header", foreground="#9b59b6", font=(*UI_THEME["font_mission_control"], "bold"))
        log_display.tag_config("status", foreground="#f39c12")
        log_display.tag_config("thought", foreground="#3498db")
        log_display.tag_config("result", foreground="#2ecc71")
        log_display.tag_config("llm", foreground="#7f8c8d", font=(*UI_THEME["font_mission_control"], "italic"))
        log_display.tag_config("error", foreground="#e74c3c", font=(*UI_THEME["font_mission_control"], "bold"))
        
        ui_queue = queue.Queue()
        stop_event = Event()

        def stop_agent(): stop_event.set(); stop_button.config(state=tk.DISABLED, text="Stopping...")
        
        stop_button = tk.Button(win, text="STOP AGENT", command=stop_agent, bg=UI_THEME["bg_button_danger"], fg="white")
        stop_button.pack(pady=5)
        
        agent_instance = AutonomousAgent(goal, self.agent.api_client, self.agent.code_interpreter, self.agent.cfg, ui_queue, stop_event)
        Thread(target=agent_instance.run, daemon=True).start()

        def process_queue():
            try:
                while not ui_queue.empty():
                    msg = ui_queue.get_nowait()
                    log_display.config(state=tk.NORMAL)
                    log_display.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg['content']}\n", (msg['tag'],))
                    log_display.config(state=tk.DISABLED)
                    log_display.see(tk.END)
                if not stop_event.is_set(): win.after(200, process_queue)
                else: stop_button.config(text="AGENT STOPPED")
            except Exception: pass # Window closed
        
        win.after(200, process_queue)

# ----------------------------------------------------------------------------
# App entrypoint
# ----------------------------------------------------------------------------
def main():
    if not ASYNC_MODE: logger.warning("aiohttp not installed. Running in sync mode.")
    if sys.platform == "win32" and ASYNC_MODE: asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app = CatGPTFusion()
    app.mainloop()

if __name__ == "__main__":
    main()
