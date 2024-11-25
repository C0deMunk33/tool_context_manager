import ast
import json
import inspect
import importlib.util
import os
from typing import Dict, Any, List
import sys
import asyncio
import os
import json
import shutil
from urllib.parse import urlparse
import aiohttp
import asyncio
from git import Repo
import nest_asyncio
import asyncio

from common import generate_grammar, generate_format_description

import subprocess
import os
from typing import Optional

class PoetryDependencyManager:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
    def _check_poetry_installed(self) -> bool:
        """Check if Poetry is installed on the system."""
        try:
            subprocess.run(['poetry', '--version'], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
            
    def _install_poetry(self) -> bool:
        """Install Poetry using the official installer."""
        try:
            subprocess.run(
                ['curl', '-sSL', 'https://install.python-poetry.org', '-o', 'install-poetry.py'],
                check=True
            )
            subprocess.run(['python', 'install-poetry.py'], check=True)
            os.remove('install-poetry.py')
            return True
        except subprocess.CalledProcessError:
            return False

    def _get_python_version(self) -> str:
        """Get the current Python version."""
        version = sys.version_info
        return f"python{version.major}.{version.minor}"

    def setup_virtual_environment(self) -> Optional[str]:
        """Set up a Poetry virtual environment for the repository."""
        if not os.path.exists(os.path.join(self.repo_path, 'pyproject.toml')):
            print("No pyproject.toml found in repository")
            return None

        if not self._check_poetry_installed():
            print("Poetry not found, attempting to install...")
            if not self._install_poetry():
                print("Failed to install Poetry")
                return None

        try:
            # Change to repo directory
            original_dir = os.getcwd()
            os.chdir(self.repo_path)

            # Install dependencies
            subprocess.run(['poetry', 'install', '--no-root'], check=True)

            # Get virtual environment path
            result = subprocess.run(
                ['poetry', 'env', 'info', '--path'],
                check=True,
                capture_output=True,
                text=True
            )
            venv_path = result.stdout.strip()

            # Add site-packages to Python path
            python_version = self._get_python_version()
            site_packages = os.path.join(venv_path, 'lib', python_version, 'site-packages')
            
            if os.path.exists(site_packages):
                if site_packages not in sys.path:
                    sys.path.insert(0, site_packages)
                print(f"Added {site_packages} to Python path")
            else:
                # Try alternative path structure for some systems
                site_packages = os.path.join(venv_path, 'Lib', 'site-packages')  # Windows-style path
                if os.path.exists(site_packages):
                    if site_packages not in sys.path:
                        sys.path.insert(0, site_packages)
                    print(f"Added {site_packages} to Python path")
                else:
                    print(f"Warning: Could not find site-packages directory in {venv_path}")

            os.chdir(original_dir)
            return venv_path

        except subprocess.CalledProcessError as e:
            print(f"Failed to set up Poetry environment: {e}")
            os.chdir(original_dir)
            return None


class ToolContext:
    def __init__(self, context_name: str, files: List[str], tag: str = "external"):
        self.tagged_functions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.initialized_classes: Dict[str, Any] = {}
        self.init_functions: Dict[str, Dict[str, Any]] = {}
        self.class_methods: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.context_name = context_name
        for file in files:
            self.load_tagged_functions(file, tag)

    async def initialize(self):
        """Separate async initialization method"""
        print("\nDEBUG: Starting initialization")
        files_to_process = set()
        
        # Collect all files from both tagged functions and class methods
        for funcs in self.tagged_functions.values():
            for info in funcs.values():
                files_to_process.add(info['file'])
                
        for class_methods in self.class_methods.values():
            for method_info in class_methods.values():
                files_to_process.add(method_info['file'])
                
        for file in files_to_process:
            print(f"\nDEBUG: Processing file {file}")
            await self.initialize_classes(file)
            
        print("DEBUG: Finished initialization\n")
        
        # Print final loaded functions
        print("\nLoaded functions and methods:")
        for name, funcs in self.tagged_functions.items():
            for file_path, info in funcs.items():
                class_name = info.get('class_name', '')
                if class_name:
                    print(f"  * {name} [{class_name}]")
                else:
                    print(f"  * {name}")

    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        if not docstring:
            return {}
        try:
            return json.loads(docstring.strip())
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in docstring")
            print(docstring)
            return {}

    def add_parent_refs(self, tree: ast.AST) -> None:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent

    def is_class_method(self, node: ast.FunctionDef) -> bool:
        parent = getattr(node, 'parent', None)
        while parent is not None:
            if isinstance(parent, ast.ClassDef):
                return True
            parent = getattr(parent, 'parent', None)
        return False

    def get_class_name(self, node: ast.FunctionDef) -> str:
        parent = getattr(node, 'parent', None)
        while parent is not None:
            if isinstance(parent, ast.ClassDef):
                return parent.name
            parent = getattr(parent, 'parent', None)
        return None

    def load_module_from_file(self, file_path: str):
        try:
            directory = os.path.dirname(file_path)
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            if directory not in sys.path:
                sys.path.insert(0, directory)
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            print(f"Error loading module from {file_path}: {str(e)}")
            return None
        finally:
            if directory in sys.path:
                sys.path.remove(directory)

    def load_tagged_functions(self, file_path: str, tag: str = "external") -> None:
        """Load all tagged functions and class methods from a file."""
        print(f"\nDEBUG: Loading tagged functions from {file_path}")
        module = self.load_module_from_file(file_path)
        if not module:
            print(f"DEBUG: Failed to load module from {file_path}")
            return

        with open(file_path, 'r') as file:
            tree = ast.parse(file.read())
            
        self.add_parent_refs(tree)
        current_class = None

        # First pass: collect classes and init functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                print(f"DEBUG: Processing class {node.name}")
                current_class = node.name
                
                # Look for init-decorated functions that return this class
                for init_func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                    if any(d.id == 'init' for d in init_func.decorator_list if isinstance(d, ast.Name)):
                        print(f"DEBUG: Found {'async ' if isinstance(init_func, ast.AsyncFunctionDef) else ''}init function {init_func.name} for class {current_class}")
                        try:
                            func = getattr(module, init_func.name)
                            self.init_functions[init_func.name] = {
                                'function': func,
                                'file': file_path,
                                'class_name': current_class,
                                'is_async': isinstance(init_func, ast.AsyncFunctionDef)
                            }
                        except AttributeError:
                            print(f"Warning: Could not load init function {init_func.name}")

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(d.id == tag for d in node.decorator_list if isinstance(d, ast.Name)):
                    is_method = self.is_class_method(node)
                    class_name = self.get_class_name(node) if is_method else None
                    
                    try:
                        if is_method:
                            print(f"DEBUG: Found {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}class method {node.name} in class {class_name}")
                            docstring = ast.get_docstring(node)
                            parsed_docstring = self.parse_docstring(docstring)
                            
                            if class_name not in self.class_methods:
                                self.class_methods[class_name] = {}
                            self.class_methods[class_name][node.name] = {
                                'file': file_path,
                                'docstring': docstring,
                                'parsed_docstring': parsed_docstring,
                                'is_async': isinstance(node, ast.AsyncFunctionDef)
                            }
                        else:
                            print(f"DEBUG: Found {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}standalone function {node.name}")
                            func = getattr(module, node.name)
                            docstring = ast.get_docstring(node)
                            parsed_docstring = self.parse_docstring(docstring)

                            if node.name not in self.tagged_functions:
                                self.tagged_functions[node.name] = {}
                            
                            self.tagged_functions[node.name][file_path] = {
                                'function': func,
                                'signature': str(inspect.signature(func)),
                                'details': parsed_docstring,
                                'file': file_path,
                                'is_async': isinstance(node, ast.AsyncFunctionDef)
                            }
                            print(f"DEBUG: Loaded function {node.name}")
                    except AttributeError as e:
                        print(f"Warning: Could not load function/method {node.name}: {e}")

    async def initialize_classes(self, file_path: str) -> None:
        """Initialize any classes that have init-tagged functions."""
        print(f"\nDEBUG: initialize_classes for {file_path}")
        print(f"DEBUG: Found init functions: {list(self.init_functions.keys())}")
        
        for init_func_name, init_func in self.init_functions.items():
            if init_func['file'] == file_path:
                try:
                    print(f"\nDEBUG: Initializing {init_func_name}")
                    func = init_func['function']
                    is_async = init_func.get('is_async', False)
                    print(f"DEBUG: Function is async: {is_async}")
                    
                    if is_async:
                        instance = await func()
                    else:
                        instance = func()
                    
                    print(f"DEBUG: Instance created: {instance}")
                    
                    class_name = init_func['class_name']
                    print(f"DEBUG: Class name: {class_name}")
                    
                    if class_name:
                        self.initialized_classes[class_name] = instance
                        print(f"DEBUG: About to load class methods for {class_name}")
                        await self.load_class_methods(instance, file_path)
                        print(f"DEBUG: Finished loading class methods for {class_name}")
                except Exception as e:
                    print(f"ERROR in initialize_classes: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
    
    async def load_class_methods(self, instance: Any, file_path: str) -> None:
        """Load class methods after class initialization."""
        print(f"\nDEBUG: load_class_methods for {instance.__class__.__name__}")
        
        for attr_name in dir(instance.__class__):
            if attr_name.startswith('__'):
                continue
                
            try:
                attr = getattr(instance.__class__, attr_name)
                print(f"DEBUG: Checking method {attr_name}")
                print(f"DEBUG: Has _external_tagged: {hasattr(attr, '_external_tagged')}")
                
                if hasattr(attr, '_external_tagged') and getattr(attr, '_external_tagged', False):
                    print(f"DEBUG: Found tagged method {attr_name}")
                    bound_method = getattr(instance, attr_name)
                    docstring = attr.__doc__
                    parsed_docstring = self.parse_docstring(docstring)
                    
                    if attr_name not in self.tagged_functions:
                        self.tagged_functions[attr_name] = {}
                    
                    sig = inspect.signature(attr)
                    parameters = list(sig.parameters.values())[1:]  # Skip 'self'
                    new_sig = sig.replace(parameters=parameters)
                    
                    self.tagged_functions[attr_name][file_path] = {
                        'function': bound_method,
                        'signature': str(new_sig),
                        'details': parsed_docstring,
                        'file': file_path,
                        'class_name': instance.__class__.__name__
                    }
                    print(f"DEBUG: Successfully loaded method {attr_name}")
            except Exception as e:
                print(f"ERROR in load_class_methods for {attr_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                   
    def get_function_list_text(self) -> str:
        """Get a text representation of all loaded functions and class methods."""
        text = "Available functions:\n"
        
        # List all functions (both standalone and class methods)
        for func_name, func_info in self.tagged_functions.items():
            for file_path, details in func_info.items():
                if isinstance(details['details'], str):
                    parsed_details = self.parse_docstring(details['details'])
                else:
                    parsed_details = details['details']
                
                description = parsed_details.get('description', 'No description available')
                class_name = details.get('class_name', '')
                if class_name:
                    text += f"    * {func_name} [{class_name}]: {description}\n"
                else:
                    text += f"    * {func_name}: {description}\n"

        text += "    * null: Exit the function context, unless you really need to use a function, err on the side of not.\n"
        return text

    def get_function(self, func_name: str, file_path: str = None) -> Dict[str, Any]:
        """Get function details by name and optional file path."""
        if func_name == "null":
            return None
            
        if func_name in self.tagged_functions:
            if file_path is None:
                if len(self.tagged_functions[func_name]) > 1:
                    raise ValueError(f"Multiple functions named '{func_name}' found. Please specify a file path.")
                file_path = next(iter(self.tagged_functions[func_name]))
                
            if file_path in self.tagged_functions[func_name]:
                return self.tagged_functions[func_name][file_path]
                
        raise ValueError(f"Function '{func_name}' not found")

    def is_async_function(self, func: Any) -> bool:
        """Check if a function is asynchronous."""
        return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)

    async def run_function(self, func_name: str, file_path: str = None, **kwargs):
        """Run a function by name with provided arguments, handling both sync and async functions."""
        context = self.get_current_context()
        func_info = context.get_function(func_name, file_path)
        if func_info is None:
            return None

        func = func_info['function']
        if context.is_async_function(func):
            return await func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, **kwargs)

    # TODO: add support for json schema

    def get_function_grammar(self, func_name: str) -> str:
        """Get grammar for a specific function."""
        if func_name == "null":
            return "null"
            
        # Check both regular functions and class methods
        function_sources = [self.tagged_functions, self.class_methods]
        for source in function_sources:
            if func_name in source:
                # Get the first file path if none specified
                file_path = next(iter(source[func_name]))
                func_info = source[func_name][file_path]
                
                if isinstance(func_info['details'], str):
                    details = json.loads(func_info['details'])
                else:
                    details = func_info['details']
                
                if 'args' in details:                
                    args_list = [{"name": arg['name'], "type": arg['type']} for arg in details['args']]
                    return generate_grammar(args_list)
                else:
                    args_list = []
                    return generate_grammar(args_list)
        
        return f"Function '{func_name}' not found."

    def get_function_format_description(self, func_name: str) -> str:
        """Get format description for function arguments."""
        if func_name == "null":
            return "No arguments needed for null function"
            
        # Check both regular functions and class methods
        function_sources = [self.tagged_functions, self.class_methods]
        for source in function_sources:
            if func_name in source:
                # Get the first file path if none specified
                file_path = next(iter(source[func_name]))
                func_info = source[func_name][file_path]
                print(f"DEBUG: Found function info for {func_name}")
                print(func_info)
                
                if isinstance(func_info['details'], str):
                    details = json.loads(func_info['details'])
                else:
                    details = func_info['details']
                
                if 'args' not in details:
                    return "No argument format description available"
                
                return generate_format_description(details['args'])
        
        return f"Function '{func_name}' not found"

class ToolContextManager:
    def __init__(self):
        self.function_contexts = {}
        self.current_context = None
        self.repo_dir = "repos"

    async def create_function_context(self, context_name: str, files: List[str], tag: str = "external", autoload: bool = True):
        print(f"\nInitializing context: {context_name}")
        if context_name in self.function_contexts:
            raise ValueError(f"Context {context_name} already exists.")
        context = ToolContext(context_name, files, tag)
        
        # Initialize BEFORE printing function list
        await context.initialize()
        
        self.function_contexts[context_name] = context
        if self.current_context is None or autoload:
            self.current_context = context_name
        
        print(f"Added function context for {context_name}")
        print(context.get_function_list_text())

    async def create_function_context_from_github(self, context_name: str, repo: str, tag: str = "external", autoload: bool = True):
        if context_name in self.function_contexts:
            raise ValueError(f"Context {context_name} already exists.")

        # Ensure repos directory exists
        os.makedirs(self.repo_dir, exist_ok=True)
        
        # Parse repository information
        if "github.com" in repo:
            parsed = urlparse(repo)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                owner, repo_name = path_parts[:2]
            else:
                raise ValueError("Invalid GitHub repository URL")
        else:
            parts = repo.split('/')
            if len(parts) != 2:
                raise ValueError("Repository should be in format 'owner/repo'")
            owner, repo_name = parts

        repo_path = os.path.join(self.repo_dir, f"{owner}_{repo_name}")

        print(f"\nInitializing context {context_name} from GitHub repository {owner}/{repo_name}")

        # Clean up existing repo directory if it exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        # Clone the repository
        try:
            clone_url = f"https://github.com/{owner}/{repo_name}.git"
            Repo.clone_from(clone_url, repo_path)
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {str(e)}")

        # Handle dependencies before loading config
        dependency_manager = PoetryDependencyManager(repo_path)
        venv_path = dependency_manager.setup_virtual_environment()
        
        if not venv_path:
            print("Warning: Failed to set up Poetry environment, some dependencies may be missing")

        # Look for config.json
        config_path = os.path.join(repo_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError("Repository must contain a config.json file")

        # Read config.json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read config.json: {str(e)}")

        # Get tool files from config
        tool_files = config.get("tool_files")
        if not tool_files:
            raise ValueError("config.json must specify an 'tool_files' field")

        files_to_add = []

        for tool_file in tool_files:
            file_path = os.path.join(repo_path, tool_file)
            if not os.path.exists(file_path):
                raise ValueError(f"Tool file {tool_file} not found in repository")
            files_to_add.append(file_path)

        # Create context
        context = ToolContext(context_name, files_to_add, tag)

        # Initialize BEFORE printing function list
        await context.initialize()

        self.function_contexts[context_name] = context
        if self.current_context is None or autoload:
            self.current_context = context_name

        print(f"Added function context for {context_name}")
        print(context.get_function_list_text())



    async def add_to_current_context(self, files: List[str], tag: str = "external"):
        if self.current_context is None:
            raise ValueError("No current function context set.")
        context = self.get_current_context()
        for file in files:
            context.load_tagged_functions(file, tag)
        await context.initialize()
        print(context.get_function_list_text())

    async def add_to_current_context_from_github(self, repo: str, tag: str = "external"):
        if self.current_context is None:
            raise ValueError("No current function context set.")

        # Ensure repos directory exists
        os.makedirs(self.repo_dir, exist_ok=True)
        
        # Parse repository information
        if "github.com" in repo:
            parsed = urlparse(repo)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                owner, repo_name = path_parts[:2]
            else:
                raise ValueError("Invalid GitHub repository URL")
        else:
            parts = repo.split('/')
            if len(parts) != 2:
                raise ValueError("Repository should be in format 'owner/repo'")
            owner, repo_name = parts

        repo_path = os.path.join(self.repo_dir, f"{owner}_{repo_name}")

        # Clean up existing repo directory if it exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        # Clone the repository
        try:
            clone_url = f"https://github.com/{owner}/{repo_name}.git"
            Repo.clone_from(clone_url, repo_path)
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {str(e)}")

        # Look for config.json
        config_path = os.path.join(repo_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError("Repository must contain a config.json file")

        # Read config.json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read config.json: {str(e)}")

        # Get entry point from config
        tool_files = config.get("tool_files")
        if not tool_files:
            raise ValueError("config.json must specify an 'tool_files' field")

        files_to_add = []

        for tool_file in tool_files:
            file_path = os.path.join(repo_path, tool_file)
            if not os.path.exists(file_path):
                raise ValueError(f"Tool file {tool_file} not found in repository")

            files_to_add.append(file_path)

        # Add file to current context
        await self.add_to_current_context(files_to_add, tag)

        print(f"Successfully loaded functions from {owner}/{repo_name}")

    def set_current_context(self, context_name: str):
        if context_name not in self.function_contexts:
            # create empty context if it doesn't exist
            self.function_contexts[context_name] = ToolContext(context_name, [])
        self.current_context = context_name

    def get_current_context(self) -> ToolContext:
        if self.current_context is None:
            raise ValueError("No current function context set.")
        return self.function_contexts[self.current_context]
    
    def get_context_names(self) -> List[str]:
        return list(self.function_contexts.keys())
    
    def get_current_context_name(self) -> str:
        return self.current_context
    
    def get_current_context_function_list_text(self) -> str:
        return self.get_current_context().get_function_list_text()
    
    def get_current_context_function_grammar(self, func_name: str) -> str:
        return self.get_current_context().get_function_grammar(func_name)
    
    def get_current_arg_format_description(self, func_name: str) -> str:
        return self.get_current_context().get_function_format_description(func_name)
    
    def get_current_function_list_grammar_enum(self) -> str:
        grammar_enum = "("
        context = self.get_current_context()
        for func_name in context.tagged_functions.keys():
            grammar_enum += f"\"{func_name}\" |"
        grammar_enum = grammar_enum + " null )"
        return grammar_enum
    
    def get_function(self, func_name: str, file_path: str = None) -> Dict[str, Any]:
        return self.get_current_context().get_function(func_name, file_path)
    
    async def run_function(self, func_name: str, file_path: str = None, **kwargs):
        """Run a function by name with provided arguments, handling both sync and async functions."""
        context = self.get_current_context()
        func_info = context.get_function(func_name, file_path)
        if func_info is None:
            return None

        func = func_info['function']
        
        loop = asyncio.get_event_loop()

        if context.is_async_function(func):
            return await func(**kwargs)
        else:
            return await loop.run_in_executor(None, func, **kwargs)

    def add_function(self, func_pointer):
        # get file, signature, and details from the function pointer
        file = inspect.getfile(func_pointer)
        signature = str(inspect.signature(func_pointer))
        details = func_pointer.__doc__
        # get the function name
        func_name = func_pointer.__name__

        # add the function to the current context
        self.get_current_context().tagged_functions[func_name] = {
            file: {
                'function': func_pointer,
                'signature': signature,
                'details': details,
                'file': file
            }
        }


nest_asyncio.apply()

# Example usage
async def main():
    
    test_repo = "https://github.com/C0deMunk33/discord_agent_interface/"

    fcm = ToolContextManager()

    await fcm.create_function_context_from_github("test", test_repo, tag="external", autoload=True)
    
if __name__ == "__main__":
    asyncio.run(main())