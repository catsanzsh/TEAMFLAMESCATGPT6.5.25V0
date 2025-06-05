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
from datetime import datetime
from pathlib import Path
from threading import Thread
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
MEMORY_FILE = ARCHIVE_DIR / "memory.json"
MODEL_FILE = ARCHIVE_DIR / "models.json"
ARCHIVE_INDEX = ARCHIVE_DIR / "archive_index.json"

# Ensure directories exist
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── Constants ──────────────────────────────────
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY" # Environment variable name
DEFAULT_MODELS = ["meta-llama/llama-3-8b-instruct", "gpt-3.5-turbo", "claude-3-opus", "gpt-4"]
LLM_TIMEOUT = 120  # seconds
CODE_EXEC_TIMEOUT = 30 # Timeout for sandboxed code execution

# UI Theme elements
UI_THEME = {
    "bg_primary": "#f5f5f5", "bg_secondary": "#ffffff", "bg_tertiary": "#f0e6ff",
    "bg_chat_display": "#fafafa", "bg_chat_input": "#f9f9f9", "bg_editor": "#1e2838",
    "bg_editor_header": "#34495e", "bg_button_primary": "#10a37f", "bg_button_secondary": "#3498db",
    "bg_button_danger": "#e74c3c", "bg_button_warning": "#f39c12", "bg_button_evolution": "#9b59b6",
    "bg_button_evo_compile": "#27ae60", "bg_button_info": "#5dade2", "bg_listbox_select": "#6c5ce7",
    "fg_primary": "#2c3e50", "fg_secondary": "#ecf0f1", "fg_button_light": "#ffffff",
    "fg_evolution_header": "#6c5ce7", "font_default": ("Consolas", 11), "font_chat": ("Consolas", 11),
    "font_button_main": ("Arial", 11, "bold"), "font_button_small": ("Arial", 10),
    "font_title": ("Arial", 14, "bold"), "font_editor": ("Consolas", 11), "font_listbox": ("Consolas", 9),
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

# ----------------------------------------------------------------------------
# Optimized API Client
# ----------------------------------------------------------------------------
class APIClient:
    """
    Handles all communication with the LLM API, managing sessions efficiently.
    """
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key_getter: Callable[[], str], timeout: int):
        self._api_key_getter = api_key_getter
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Initializes and returns a single aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_headers(self) -> Dict[str, str]:
        """Constructs the authorization headers."""
        api_key = self._api_key_getter()
        if not api_key:
            raise RuntimeError("API Key is missing. Please configure it.")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _parse_response(self, data: Dict[str, Any]) -> str:
        """Safely extracts the message content from the API response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected LLM API response structure: {data}. Error: {e}")
            raise RuntimeError("Invalid response structure from LLM API.")

    async def call_async(self, payload: Dict[str, Any]) -> str:
        """Fires a single completion request asynchronously."""
        session = await self._get_session()
        try:
            async with session.post(self.API_URL, headers=self._get_headers(), json=payload, timeout=self.timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"LLM API call failed with status {resp.status}: {error_text}")
                    raise RuntimeError(f"API Error (Status {resp.status}): {error_text}")
                data = await resp.json()
                return self._parse_response(data)
        except aiohttp.ClientError as e:
            logger.error(f"AIOHTTP client error during LLM call: {e}")
            raise RuntimeError(f"Network Error: {e}")

    def call_sync(self, payload: Dict[str, Any]) -> str:
        """Fires a single completion request synchronously."""
        try:
            response = requests.post(self.API_URL, headers=self._get_headers(), json=payload, timeout=self.timeout)
            if response.status_code != 200:
                logger.error(f"LLM API call failed with status {response.status_code}: {response.text}")
                raise RuntimeError(f"API Error (Status {response.status_code}): {response.text}")
            return self._parse_response(response.json())
        except requests.RequestException as e:
            logger.error(f"Requests error during LLM call: {e}")
            raise RuntimeError(f"Network Error: {e}")

    async def close_session(self):
        """Closes the aiohttp session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("AIOHTTP session closed.")

# ----------------------------------------------------------------------------
# Code Interpreter for Sandboxed Execution
# ----------------------------------------------------------------------------
class CodeInterpreter:
    """Executes Python code in an isolated, sandboxed environment."""
    def __init__(self, timeout: int = CODE_EXEC_TIMEOUT):
        self.timeout = timeout
        self.exec_base_dir = ARCHIVE_DIR / "code_execution_sandbox"
        self.exec_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Code interpreter will use base execution directory: {self.exec_base_dir}")

    def execute_code(self, code_string: str) -> Tuple[str, str, Optional[str]]:
        """
        Executes a string of Python code in a subprocess.
        Returns: (stdout, stderr, error_message)
        """
        run_id = now_ts()
        current_run_dir = self.exec_base_dir / run_id
        current_run_dir.mkdir(parents=True, exist_ok=True)
        temp_script_path = current_run_dir / "temp_script.py"
        
        stdout_str, stderr_str, error_msg = "", "", None

        try:
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(code_string)

            process = subprocess.run(
                [sys.executable, "-u", str(temp_script_path)], # Use str() for path
                capture_output=True, text=True, timeout=self.timeout,
                cwd=str(current_run_dir), check=False
            )
            stdout_str, stderr_str = process.stdout, process.stderr
            if process.returncode != 0:
                logger.warning(f"Code execution finished with non-zero exit code: {process.returncode}")

        except subprocess.TimeoutExpired:
            error_msg = f"Code execution timed out after {self.timeout} seconds."
            stderr_str += f"\nTimeoutError: Execution exceeded {self.timeout} seconds and was terminated."
        except Exception as e:
            error_msg = f"An unexpected error occurred during subprocess execution: {e}"
            stderr_str += f"\nSandboxSetupError: {e}"
        finally:
            # Cleanup is important
            if current_run_dir.exists():
                shutil.rmtree(current_run_dir, ignore_errors=True)
        
        return stdout_str, stderr_str, error_msg

# ----------------------------------------------------------------------------
# DarwinAgent – Refactored for maintainability
# ----------------------------------------------------------------------------
class DarwinAgent:
    """LLM agent that can fork, load, evaluate child agents, and execute Python code."""

    def __init__(self):
        self.models: List[str] = self._load_models()
        self.cfg: Dict[str, Any] = {
            "model": self.models[0] if self.models else DEFAULT_MODELS[0],
            "temperature": 0.7, "max_tokens": 4096, "top_p": 0.9,
            "frequency_penalty": 0.0, "presence_penalty": 0.0,
        }
        self.history: List[Dict[str, str]] = self._load_memory()
        self.agent_archive: List[Tuple[str, str, str, Optional[str]]] = self._load_agent_archive()
        self.plugins: Dict[str, ModuleType] = {}
        self._discover_plugins()
        self.api_client = APIClient(get_api_key, LLM_TIMEOUT)
        self.code_interpreter = CodeInterpreter(timeout=CODE_EXEC_TIMEOUT)
        logger.info("DarwinAgent initialized.")

    # --- File I/O Helpers ---
    def _load_json_file(self, file_path: Path, default: Union[List, Dict]) -> Union[List, Dict]:
        if not file_path.exists(): return default
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading {file_path}: {e}. Returning default.")
            return default

    def _save_json_file(self, file_path: Path, data: Union[List, Dict]):
        try:
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except IOError as e:
            logger.error(f"Failed to save {file_path}: {e}")

    def _load_memory(self): return self._load_json_file(MEMORY_FILE, [])
    def _save_memory(self): self._save_json_file(MEMORY_FILE, self.history[-2000:])
    def _load_models(self):
        models = self._load_json_file(MODEL_FILE, [])
        if not models:
            self._save_json_file(MODEL_FILE, DEFAULT_MODELS)
            return list(DEFAULT_MODELS)
        return models
    def _load_agent_archive(self): return self._load_json_file(ARCHIVE_INDEX, [])
    def _save_agent_archive(self): self._save_json_file(ARCHIVE_INDEX, self.agent_archive)

    # --- Plugin Management ---
    def _discover_plugins(self):
        if str(PLUGIN_DIR) not in sys.path:
            sys.path.insert(0, str(PLUGIN_DIR))
        loaded_plugins = {}
        for py_file in PLUGIN_DIR.glob("*.py"):
            name = py_file.stem
            if name == "__init__": continue
            try:
                spec = importlib.util.spec_from_file_location(name, py_file)
                if not spec or not spec.loader:
                    logger.warning(f"Could not create spec for plugin {name} at {py_file}")
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "run"):
                    loaded_plugins[name] = mod
                    logger.info(f"Plugin '{name}' loaded successfully.")
                else:
                    logger.warning(f"Plugin {name} loaded but has no 'run' method.")
            except Exception as e:
                logger.error(f"Plugin '{name}' failed to load: {e}\n{traceback.format_exc()}")
        self.plugins = loaded_plugins

    async def _run_plugin_async(self, name: str, args: str) -> str:
        plugin = self.plugins.get(name)
        if not plugin: return f"[plugin‑error] no such plugin: {name}"
        try:
            if inspect.iscoroutinefunction(plugin.run): return str(await plugin.run(args))
            return await asyncio.to_thread(plugin.run, args)
        except Exception as e: return f"[plugin‑error] {name}: {e}"

    def _run_plugin_sync(self, name: str, args: str) -> str:
        plugin = self.plugins.get(name)
        if not plugin: return f"[plugin‑error] no such plugin: {name}"
        try:
            if inspect.iscoroutinefunction(plugin.run):
                return f"[plugin‑error] {name}: cannot run async plugin synchronously"
            return str(plugin.run(args))
        except Exception as e: return f"[plugin‑error] {name}: {e}"

    # --- Core Logic ---
    def _prepare_payload(self) -> Dict[str, Any]:
        """Prepares the payload for the LLM API call."""
        system_prompt = (
            "You are DarwinCat, a helpful AI assistant with Python code execution abilities. "
            "To execute Python code, respond with a fenced code block: ```python_exec\\n# your code here\\n```. "
            "The code runs in a sandbox, and you will see its output. Use this to solve problems. "
            "For non-executable examples, use ```python...```."
        )
        messages = list(self.history)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        return {"model": self.cfg["model"], "messages": messages[-20:], "temperature": float(self.cfg["temperature"]),
                "max_tokens": int(self.cfg["max_tokens"]), "top_p": float(self.cfg["top_p"]), 
                "frequency_penalty": float(self.cfg["frequency_penalty"]), "presence_penalty": float(self.cfg["presence_penalty"])}

    def _extract_executable_code(self, message: str) -> Optional[str]:
        """Extracts code from ```python_exec ... ``` blocks."""
        match = re.search(r"```python_exec\s*\n(.*?)\n```", message, re.DOTALL)
        return match.group(1).strip() if match else None
        
    def _format_code_execution_summary(self, stdout: str, stderr: str, exec_error: Optional[str]) -> str:
        """Creates a formatted summary of the code execution results."""
        summary = "--- Python Code Execution ---\n"
        has_output = False
        if stdout:
            summary += f"STDOUT:\n```text\n{stdout.strip()}\n```\n"
            has_output = True
        if stderr:
            summary += f"STDERR:\n```text\n{stderr.strip()}\n```\n"
            has_output = True
        if exec_error:
            summary += f"EXECUTION SYSTEM ERROR: {exec_error}\n"
            has_output = True
        if not has_output:
            summary += "Code executed successfully with no output.\n"
        return summary.strip()

    async def _ask_orchestrator(
        self,
        user_msg: str,
        llm_caller: Callable[[Dict], Awaitable[str]],
        plugin_runner: Callable[[str, str], Awaitable[str]],
        code_executor: Callable[[str], Awaitable[Tuple[str, str, Optional[str]]]],
    ) -> str:
        """
        Centralized logic for handling user messages, calling LLMs, and executing code.
        """
        if user_msg.startswith("/model "):
            mdl = user_msg.split(maxsplit=1)[1]
            if mdl in self.models:
                self.cfg["model"] = mdl
                return f"Model switched to {mdl}"
            return f"Unknown model: {mdl}. Available: {', '.join(self.models)}"

        if user_msg.startswith("/tool "):
            _, rest = user_msg.split(maxsplit=1)
            name, *arg_tokens = rest.split(maxsplit=1)
            return await plugin_runner(name, arg_tokens[0] if arg_tokens else "")

        self.history.append({"role": "user", "content": user_msg})
        try:
            # First LLM call
            payload = self._prepare_payload()
            assistant_msg_1 = await llm_caller(payload)
            self.history.append({"role": "assistant", "content": assistant_msg_1})
            self._save_memory()

            code_to_execute = self._extract_executable_code(assistant_msg_1)
            if not code_to_execute:
                return assistant_msg_1

            # Code execution
            logger.info(f"Executing code:\n{textwrap.shorten(code_to_execute, 500)}")
            stdout, stderr, exec_error = await code_executor(code_to_execute)
            summary = self._format_code_execution_summary(stdout, stderr, exec_error)
            
            # Second LLM call with code results
            feedback_msg = f"The Python code was executed. Result:\n{summary}"
            self.history.append({"role": "user", "content": feedback_msg})
            payload_2 = self._prepare_payload()
            assistant_msg_2 = await llm_caller(payload_2)
            self.history.append({"role": "assistant", "content": assistant_msg_2})
            self._save_memory()
            
            return (f"{assistant_msg_1}\n\n{summary}\n\n"
                    f"CatGPT (after code execution):\n{assistant_msg_2}")

        except RuntimeError as e:
            return f"[LLM-Error] {e}"
        except Exception as e:
            logger.error(f"Unexpected error in orchestrator: {e}\n{traceback.format_exc()}")
            return f"[Agent-Error] An unexpected error occurred: {e}"

    async def ask_async(self, user_msg: str) -> str:
        """Handles a user request asynchronously."""
        async def code_exec_wrapper(code: str):
            return await asyncio.to_thread(self.code_interpreter.execute_code, code)
        
        return await self._ask_orchestrator(user_msg, self.api_client.call_async, self._run_plugin_async, code_exec_wrapper)

    def ask_sync(self, user_msg: str) -> str:
        """Handles a user request synchronously."""
        async def sync_llm_caller(payload): return self.api_client.call_sync(payload)
        async def sync_plugin_runner(name, args): return self._run_plugin_sync(name, args)
        async def sync_code_executor(code): return self.code_interpreter.execute_code(code)

        # We must run the orchestrator within an asyncio event loop, even for the "sync" version.
        return asyncio.run(self._ask_orchestrator(user_msg, sync_llm_caller, sync_plugin_runner, sync_code_executor))

    # --- Agent Evolution ---
    def try_agent_compile(self, path: Path, code: str) -> Tuple[str, Optional[str]]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name

            spec = importlib.util.spec_from_file_location("TestAgent", tmp_path)
            if not spec or not spec.loader:
                return "QUARANTINED", "Could not load module spec"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            agent_class = getattr(module, "DarwinAgent", None)
            if agent_class is None:
                return "QUARANTINED", "No DarwinAgent class found."
            
            return "FIT", None
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Agent compilation/test failed: {e}\n{tb}")
            return "QUARANTINED", tb
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)


    def recompile(self, new_code: str) -> Tuple[str, str, Optional[str]]:
        ts = now_ts()
        fname_stem = f"CatGPT_Agent_v{ts}"
        agent_file_path = ARCHIVE_DIR / f"{fname_stem}.py"
        
        status, error = self.try_agent_compile(agent_file_path, new_code)
        
        error_display = str(error) if error else 'None'
        final_code_header = f'''"""
Darwin CatGPT Fusion Agent - Generated by self-recompile
Timestamp: {datetime.now()}
Status: {status}
Error: {textwrap.shorten(error_display, width=100, placeholder="...")}
"""

'''
        final_code = final_code_header + new_code
        agent_file_path.write_text(final_code, encoding="utf-8")
        
        readme_path = ARCHIVE_DIR / f"README_{fname_stem}.txt"
        readme_content = f"Agent: {agent_file_path.name}\nStatus: {status}\nError: {error_display}"
        readme_path.write_text(readme_content, encoding="utf-8")
        
        self.agent_archive.append((ts, agent_file_path.name, status, str(error))) # Store error as string
        self._save_agent_archive()
        
        logger.info(f"Agent recompiled: {agent_file_path.name}, Status: {status}")
        return agent_file_path.name, status, str(error)
        
    async def shutdown(self):
        """Gracefully shuts down agent resources."""
        await self.api_client.close_session()

# ----------------------------------------------------------------------------
# Tkinter UI layer – With simplified logic
# ----------------------------------------------------------------------------
class CatGPTFusion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Darwin CatGPT Fusion Edition - Optimized")
        self.geometry("1100x750")
        self.config(bg=UI_THEME["bg_primary"])
        
        self._prompt_for_api_key_if_missing()  
        self.agent = DarwinAgent()
        self.intro_message = "Welcome to Darwin CatGPT Fusion! I can execute Python code for you."
        
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        logger.info("CatGPTFusion UI initialized.")

    def _prompt_for_api_key_if_missing(self):
        global RUNTIME_API_KEY
        env_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
        if env_key and env_key.strip():
            RUNTIME_API_KEY = env_key.strip()
            logger.info(f"API Key loaded from {OPENROUTER_API_KEY_ENV_VAR} environment variable.")
            return

        api_key_input = simpledialog.askstring(
            "API Key Required",
            f"{OPENROUTER_API_KEY_ENV_VAR} not found.\nPlease enter your OpenRouter API Key:",
            parent=self  
        )
        if api_key_input and api_key_input.strip():
            RUNTIME_API_KEY = api_key_input.strip()
            logger.info("API Key set from user prompt for this session.")
        else:
            RUNTIME_API_KEY = "" 
            messagebox.showwarning("API Key Missing", "No API Key was entered. AI features may not work.", parent=self)

    def _build_ui(self):
        main_container = tk.Frame(self, bg=UI_THEME["bg_primary"])
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_container, bg=UI_THEME["bg_secondary"], relief=tk.RAISED, bd=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        right_frame = tk.Frame(main_container, bg=UI_THEME["bg_tertiary"], relief=tk.RIDGE, bd=2)
        right_frame.pack(side="right", fill="y", padx=(5, 0))

        # Build contents of left and right panels
        self._build_chat_display(left_frame)
        self._build_input_area(left_frame)
        self._build_control_buttons(left_frame)
        self._build_archive_panel(right_frame)

        self._display_initial_messages()

    def _build_chat_display(self, parent):
        self.chat_window = scrolledtext.ScrolledText(
            parent, width=75, height=32, bg=UI_THEME["bg_chat_display"], fg=UI_THEME["fg_primary"],
            font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1
        )
        self.chat_window.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_window.configure(state=tk.DISABLED)

    def _build_input_area(self, parent):
        input_frame = tk.Frame(parent, bg=UI_THEME["bg_secondary"])
        input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.input_field = tk.Text(
            input_frame, width=60, height=3, bg=UI_THEME["bg_chat_input"], fg=UI_THEME["fg_primary"],
            font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1, insertbackground=UI_THEME["fg_primary"]
        )
        self.input_field.pack(side="left", fill="x", expand=True)
        self.input_field.bind("<Return>", lambda e: self._on_send() if not (e.state & 0x1) else None)

        send_btn = tk.Button(
            input_frame, text="Send", command=self._on_send, bg=UI_THEME["bg_button_primary"], 
            fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_main"], padx=15, relief=tk.RAISED, bd=1
        )
        send_btn.pack(side="right", padx=(5, 0), fill="y")
        
    def _build_control_buttons(self, parent):
        control_frame = tk.Frame(parent, bg=UI_THEME["bg_secondary"])
        control_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        buttons_config = [
            ("Model Config", self._model_config_window, UI_THEME["bg_button_secondary"]),
            ("Agent Recompile", self._agent_recompile_window, UI_THEME["bg_button_evolution"]),
            ("Plugin Manager", self._plugin_manager, UI_THEME["bg_button_warning"]),
            ("Copy Sandbox Code", self._copy_codebase_to_clipboard, UI_THEME["bg_button_info"]),
            ("Clear History", self._clear_history, UI_THEME["bg_button_danger"]),
        ]
        for text, command, bg_color in buttons_config:
            tk.Button(
                control_frame, text=text, command=command, bg=bg_color, fg=UI_THEME["fg_button_light"], 
                font=UI_THEME["font_button_small"], padx=10, relief=tk.RAISED, bd=1, cursor="hand2"
            ).pack(side="left", padx=2, pady=2)

    def _build_archive_panel(self, parent):
        tk.Label(
            parent, text="🧬 Evolution Archive", font=UI_THEME["font_title"], 
            bg=UI_THEME["bg_tertiary"], fg=UI_THEME["fg_evolution_header"]
        ).pack(pady=8)
        self._build_archive_listbox(parent)
        self._build_archive_buttons(parent)
        self.stats_label = tk.Label(
            parent, text="", font=("Consolas", 10), bg=UI_THEME["bg_tertiary"], fg=UI_THEME["fg_primary"]
        )
        self.stats_label.pack(pady=5)
        self._refresh_archive_listbox()

    def _build_archive_listbox(self, parent):
        list_frame = tk.Frame(parent, bg=UI_THEME["bg_tertiary"])
        list_frame.pack(fill="both", expand=True, padx=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.archive_listbox = tk.Listbox(
            list_frame, width=50, height=25, font=UI_THEME["font_listbox"], bg=UI_THEME["bg_secondary"], 
            fg=UI_THEME["fg_primary"], selectbackground=UI_THEME["bg_listbox_select"], 
            selectforeground=UI_THEME["fg_button_light"], yscrollcommand=scrollbar.set, relief=tk.SOLID, bd=1
        )
        self.archive_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.archive_listbox.yview)
    
    def _build_archive_buttons(self, parent):
        btn_frame = tk.Frame(parent, bg=UI_THEME["bg_tertiary"])
        btn_frame.pack(fill="x", padx=5, pady=5)
        tk.Button(
            btn_frame, text="Open/Edit", command=self._open_selected_agent, bg=UI_THEME["bg_listbox_select"], 
            fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"], padx=10, relief=tk.RAISED, bd=1
        ).pack(side="left", padx=2, pady=2, expand=True, fill="x")
        tk.Button(
            btn_frame, text="Delete", command=self._delete_selected, bg=UI_THEME["bg_button_danger"],
            fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"], padx=10, relief=tk.RAISED, bd=1
        ).pack(side="left", padx=2, pady=2, expand=True, fill="x")

    def _display_initial_messages(self):
        self.chat_window.configure(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"CatGPT: {self.intro_message}\n")
        self.chat_window.insert(tk.END, f"Models: {', '.join(self.agent.models)}\n")
        self.chat_window.insert(tk.END, "Commands: /model <name>, /tool <plugin> <args>\n")
        if not get_api_key():
            self.chat_window.insert(tk.END, f"\nWARNING: API Key not set. LLM features disabled.\n")
        self.chat_window.configure(state=tk.DISABLED)
    
    def _append_chat(self, who: str, txt: str):
        self.chat_window.configure(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"\n{who}:\n{txt}\n")
        self.chat_window.see(tk.END)
        self.chat_window.configure(state=tk.DISABLED)

    def _on_send(self):
        user_msg = self.input_field.get("1.0", "end-1c").strip()
        if not user_msg: return
        
        if not get_api_key() and not user_msg.startswith("/"):
            messagebox.showerror("API Key Missing", "Cannot send message. API Key not configured.", parent=self)
            return

        self.input_field.delete("1.0", tk.END)
        self._append_chat("You", user_msg)
        Thread(target=self._worker, args=(user_msg,), daemon=True).start()

    def _worker(self, msg: str):
        """Worker thread to run agent tasks without blocking the UI."""
        try:
            if ASYNC_MODE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                answer = loop.run_until_complete(self.agent.ask_async(msg))
            else:
                answer = self.agent.ask_sync(msg)
        except Exception as e:
            logger.error(f"Error in worker thread: {e}\n{traceback.format_exc()}")
            answer = f"[error] An unexpected error occurred: {e}"
        
        if self.winfo_exists():
            self.after(0, lambda: self._append_chat("CatGPT", answer))

    def _clear_history(self):
        if messagebox.askyesno("Confirm Clear", "Clear chat history and agent memory?"):
            self.agent.history.clear()
            self.agent._save_memory()
            self.chat_window.configure(state=tk.NORMAL)
            self.chat_window.delete('1.0', tk.END)
            self._display_initial_messages()
            logger.info("Chat history and agent memory cleared.")

    def _delete_selected(self):
        """Optimized: Deletes an agent using direct list indexing."""
        try:
            selected_indices = self.archive_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("No Selection", "Select an agent to delete.", parent=self)
                return

            listbox_index = selected_indices[0]
            archive_index = len(self.agent.agent_archive) - 1 - listbox_index
            
            ts, filename, _, _ = self.agent.agent_archive[archive_index]
            
            if not messagebox.askyesno("Confirm Delete", f"Permanently delete agent '{filename}'?", parent=self):
                return
            
            self.agent.agent_archive.pop(archive_index)
            self.agent._save_agent_archive()

            agent_file = ARCHIVE_DIR / filename
            readme_file = ARCHIVE_DIR / f"README_{filename.replace('.py', '.txt')}"
            
            if agent_file.exists(): agent_file.unlink()
            if readme_file.exists(): readme_file.unlink()

            logger.info(f"Deleted agent '{filename}'")
            self._refresh_archive_listbox()

        except IndexError:
            messagebox.showerror("Error", "Selected agent not found in archive. Please refresh.", parent=self)
        except Exception as e:
            logger.error(f"Error deleting agent: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Deletion Error", f"Could not delete agent: {e}", parent=self)

    def _on_closing(self):
        """Handle window closing event to shut down agent resources."""
        logger.info("Shutdown sequence initiated.")
        if ASYNC_MODE and hasattr(self, 'agent') and self.agent:
            # Ensure agent exists before trying to shut it down
            shutdown_thread = Thread(target=lambda: asyncio.run(self.agent.shutdown()), daemon=True)
            shutdown_thread.start()
        self.destroy()


    def _model_config_window(self):
        win = tk.Toplevel(self)
        win.title("Model Configuration"); win.geometry("400x350"); win.config(bg=UI_THEME["bg_secondary"])
        win.transient(self); win.grab_set()

        frame = tk.Frame(win, bg=UI_THEME["bg_secondary"], padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Model:", font=UI_THEME["font_default"], bg=UI_THEME["bg_secondary"]).grid(row=0, column=0, sticky="w", pady=5)
        model_var = tk.StringVar(value=self.agent.cfg["model"])
        model_menu = tk.OptionMenu(frame, model_var, *self.agent.models if self.agent.models else [DEFAULT_MODELS[0]])
        model_menu.config(width=25, relief=tk.SOLID, bd=1)
        model_menu.grid(row=0, column=1, pady=5, sticky="ew")

        params_config = [("Temperature:", "temperature"), ("Max Tokens:", "max_tokens"), ("Top P:", "top_p"),
                         ("Frequency Penalty:", "frequency_penalty"), ("Presence Penalty:", "presence_penalty")]
        entries: Dict[str, tk.Entry] = {}
        for i, (label_text, key) in enumerate(params_config, 1):
            tk.Label(frame, text=label_text, font=UI_THEME["font_default"], bg=UI_THEME["bg_secondary"]).grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(frame, width=20, relief=tk.SOLID, bd=1)
            entry.insert(0, str(self.agent.cfg.get(key, "")))
            entry.grid(row=i, column=1, pady=5, sticky="ew")
            entries[key] = entry

        def save_config():
            try:
                self.agent.cfg["model"] = model_var.get()
                for key, entry_widget in entries.items(): self.agent.cfg[key] = entry_widget.get()
                logger.info(f"Model configuration updated: {self.agent.cfg}")
                win.destroy()
            except Exception as e: messagebox.showerror("Error", f"Could not save configuration: {e}", parent=win)
        
        tk.Button(frame, text="Save", command=save_config, bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"]).grid(row=len(params_config)+1, columnspan=2, pady=20)

    def _agent_recompile_window(self):
        win = tk.Toplevel(self)
        win.title("Agent Recompile & Evolution"); win.geometry("900x700"); win.config(bg=UI_THEME["bg_editor"])
        win.transient(self); win.grab_set()

        header = tk.Frame(win, bg=UI_THEME["bg_editor_header"])
        header.pack(fill="x")
        tk.Label(header, text="🧬 Agent Evolution Chamber", font=("Arial", 16, "bold"), bg=UI_THEME["bg_editor_header"], fg=UI_THEME["fg_secondary"]).pack(pady=10)

        editor = tk.Text(win, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"], insertbackground=UI_THEME["bg_button_danger"], font=UI_THEME["font_editor"], undo=True)
        editor.pack(fill="both", expand=True, padx=10, pady=10)
        editor.insert(tk.END, inspect.getsource(sys.modules[__name__]))
        
        def compile_and_evolve():
            filename, status, error = self.agent.recompile(editor.get("1.0", "end-1c"))
            self._refresh_archive_listbox()
            msg = f"Agent evolved: {filename}\nStatus: {status}"
            if status != "FIT": msg += f"\nError: {error}"
            (messagebox.showinfo if status == "FIT" else messagebox.showwarning)("Evolution Result", msg, parent=win)
            win.destroy()

        tk.Button(win, text="🚀 Compile & Evolve", command=compile_and_evolve, bg=UI_THEME["bg_button_evo_compile"], fg=UI_THEME["fg_button_light"], font=("Arial", 14, "bold")).pack(pady=10)

    def _plugin_manager(self):
        win = tk.Toplevel(self)
        win.title("Plugin Manager"); win.geometry("600x400"); win.config(bg=UI_THEME["bg_secondary"])
        win.transient(self); win.grab_set()
        
        self.agent._discover_plugins()
        lst = tk.Listbox(win, font=UI_THEME["font_listbox"], relief=tk.SOLID, bd=1)
        lst.pack(fill="both", expand=True, padx=10, pady=10)
        if self.agent.plugins:
            for name in self.agent.plugins: lst.insert(tk.END, f"✓ {name}")
        else: lst.insert(tk.END, "No plugins found.")
        
        tk.Button(win, text="Refresh", command=lambda: self._plugin_manager() or win.destroy(), bg=UI_THEME["bg_button_secondary"], fg=UI_THEME["fg_button_light"]).pack(pady=10)

    def _refresh_archive_listbox(self):
        self.archive_listbox.delete(0, tk.END)
        fit_count = sum(1 for _, _, s, _ in self.agent.agent_archive if s == "FIT")
        for ts, filename, status, _ in reversed(self.agent.agent_archive):
            icon = "✅" if status == "FIT" else "🔒"
            self.archive_listbox.insert(tk.END, f"{icon} {ts.split('_')[0]} | {filename}")
        total = len(self.agent.agent_archive)
        self.stats_label.config(text=f"Total: {total} | Fit: {fit_count} | Quarantine: {total-fit_count}")

    def _open_selected_agent(self):
        try:
            listbox_index = self.archive_listbox.curselection()[0]
            archive_index = len(self.agent.agent_archive) - 1 - listbox_index
            ts, filename, status, _ = self.agent.agent_archive[archive_index]
        except IndexError:
            messagebox.showinfo("No Selection", "Select an agent to open.", parent=self)
            return

        agent_path = ARCHIVE_DIR / filename
        if not agent_path.exists():
            messagebox.showerror("File Not Found", f"Agent file not found: {filename}", parent=self)
            return

        win = tk.Toplevel(self)
        win.title(f"View/Edit: {filename} ({status})"); win.geometry("850x650"); win.config(bg=UI_THEME["bg_editor"])
        editor = tk.Text(win, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"], font=UI_THEME["font_editor"], undo=True)
        editor.pack(fill="both", expand=True, padx=5, pady=5)
        editor.insert(tk.END, agent_path.read_text(encoding="utf-8"))
        
        def save_as_new_branch():
            self.agent.recompile(editor.get("1.0", "end-1c"))
            self._refresh_archive_listbox()
            win.destroy()
        
        tk.Button(win, text="Save as New Branch", command=save_as_new_branch, bg=UI_THEME["bg_button_secondary"], fg=UI_THEME["fg_button_light"]).pack(pady=10)

    def _copy_codebase_to_clipboard(self):
        self.clipboard_clear()
        try:
            # Get the source code of the CodeInterpreter class specifically
            code_to_copy = inspect.getsource(CodeInterpreter)
            self.clipboard_append(code_to_copy)
            messagebox.showinfo("Code Copied", "Code Interpreter sandbox logic has been copied to the clipboard.", parent=self)
        except TypeError:
             messagebox.showerror("Error", "Could not retrieve the source code for the Code Interpreter.", parent=self)


# ----------------------------------------------------------------------------
# App entrypoint
# ----------------------------------------------------------------------------
def main():
    if not ASYNC_MODE:
        logger.warning("aiohttp not installed. Running in sync mode. Install with: pip install aiohttp")
    
    if sys.platform == "win32" and ASYNC_MODE:
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("Set WindowsSelectorEventLoopPolicy for asyncio.")
        except Exception as e:
            logger.warning(f"Could not set WindowsSelectorEventLoopPolicy: {e}")
    
    app = CatGPTFusion()
    app.mainloop()

if __name__ == "__main__":  
    main()
