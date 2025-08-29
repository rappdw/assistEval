"""Manual ChatGPT provider adapter for paste-based evaluation.

This module implements a manual provider that displays prompts and collects
responses via copy-paste, useful when no API is available.
"""

import signal
import subprocess
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from bench.adapters.base import Provider, ProviderError, ProviderTimeoutError


class ChatGPTManualProvider(Provider):
    """Manual ChatGPT provider for paste-based interaction.

    Displays prompts to the user and collects responses via manual paste,
    storing raw text for evaluation when API access is not available.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize manual ChatGPT provider.

        Args:
            **kwargs: Configuration options for manual interaction
        """
        super().__init__(name="ChatGPT (Manual)", **kwargs)
        self.console = Console()
        self.timeout = kwargs.get("timeout", 300)  # 5 minutes default
        self.display_format = kwargs.get("display_format", "rich")

    def invoke(
        self,
        system: str,
        user: str,
        *,
        options: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> dict[str, str]:
        """Display prompt and collect manual response.

        Args:
            system: System prompt with constraints and context
            user: User prompt with the evaluation task
            options: Manual provider options (display format, etc.)
            capabilities: Capability constraints to display

        Returns:
            Dictionary with 'raw_text' key containing manually pasted response

        Raises:
            ProviderError: If response collection fails
            ProviderTimeoutError: If user doesn't respond within timeout
        """
        try:
            # Validate capabilities first
            self.validate_capabilities(capabilities)

            # Modify system prompt with capability constraints
            modified_system = self._modify_system_prompt(system, capabilities)

            # Display the prompt to the user
            self._display_prompt(modified_system, user, capabilities)

            # Collect manual response with timeout
            response = self._collect_response()

            # Validate response is not empty
            if not response.strip():
                raise ProviderError("Empty response received")

            return {"raw_text": response}

        except KeyboardInterrupt:
            raise ProviderError("Manual input cancelled by user") from None
        except Exception as e:
            if isinstance(e, ProviderError | ProviderTimeoutError):
                raise
            raise ProviderError(f"Failed to collect manual response: {e}") from e

    def _display_prompt(
        self, system: str, user: str, capabilities: dict[str, Any]
    ) -> None:
        """Display formatted prompt to user with capability constraints.

        Args:
            system: Modified system prompt with constraints
            user: User prompt with task
            capabilities: Capability constraints to highlight
        """
        self.console.clear()

        if self.display_format == "plain":
            self._display_prompt_plain(system, user, capabilities)
        else:
            self._display_prompt_rich(system, user, capabilities)

    def _display_prompt_plain(
        self, system: str, user: str, capabilities: dict[str, Any]
    ) -> None:
        """Display plain text prompt for easy copy/paste.

        Args:
            system: Modified system prompt with constraints
            user: User prompt with task
            capabilities: Capability constraints to highlight
        """
        print("=" * 80)
        print("CHATGPT MANUAL EVALUATION")
        print("=" * 80)
        print()

        # Display capability constraints
        constraints = self._format_constraints_plain(capabilities)
        if constraints:
            print("IMPORTANT CONSTRAINTS:")
            print("-" * 20)
            print(constraints)
            print()

        # Combine system and user prompts for ChatGPT
        combined_prompt = self._combine_prompts_for_chatgpt(system, user)

        # Copy to clipboard on macOS
        clipboard_success = self._copy_to_clipboard(combined_prompt)

        print("PROMPT TO COPY:")
        print("-" * 15)
        print(combined_prompt)
        print()

        if clipboard_success:
            print("✅ Prompt copied to clipboard!")
        else:
            print("⚠️  Could not copy to clipboard - please copy manually")
        print()

        # Display instructions
        print("INSTRUCTIONS:")
        print("-" * 13)
        if clipboard_success:
            print("1. Paste the prompt into ChatGPT (already copied to clipboard)")
        else:
            print("1. Copy the prompt above and paste it into ChatGPT")
        print("2. Wait for ChatGPT's complete response")
        print("3. Copy ChatGPT's entire response")
        print(
            "4. Press Enter, and the results in the clipboard will be "
            "copied automatically"
        )
        print()

    def _display_prompt_rich(
        self, system: str, user: str, capabilities: dict[str, Any]
    ) -> None:
        """Display Rich formatted prompt with panels and colors.

        Args:
            system: Modified system prompt with constraints
            user: User prompt with task
            capabilities: Capability constraints to highlight
        """
        # Display header
        header = Text("ChatGPT Manual Evaluation", style="bold green")
        self.console.print(Panel(header, expand=False))
        self.console.print()

        # Display capability constraints prominently
        constraints = self._format_constraints(capabilities)
        if constraints:
            self.console.print(
                Panel(
                    constraints,
                    title="[red]IMPORTANT CONSTRAINTS[/red]",
                    border_style="red",
                    expand=False,
                )
            )
            self.console.print()

        # Combine system and user prompts for ChatGPT
        combined_prompt = self._combine_prompts_for_chatgpt(system, user)

        # Copy to clipboard on macOS
        clipboard_success = self._copy_to_clipboard(combined_prompt)

        # Display combined prompt
        prompt_panel = Panel(
            Syntax(combined_prompt, "text", theme="monokai", word_wrap=True),
            title="[yellow]Prompt to Copy[/yellow]",
            border_style="yellow",
        )
        self.console.print(prompt_panel)

        if clipboard_success:
            self.console.print("[green]✅ Prompt copied to clipboard![/green]")
        else:
            self.console.print(
                "[yellow]⚠️  Could not copy to clipboard - please copy manually[/yellow]"
            )
        self.console.print()

        # Display instructions
        if clipboard_success:
            instructions = Text.assemble(
                ("1. ", "bold"),
                ("Paste the prompt into ChatGPT (already copied to clipboard)\n"),
                ("2. ", "bold"),
                ("Wait for ChatGPT's complete response\n"),
                ("3. ", "bold"),
                ("Copy ChatGPT's entire response\n"),
                ("4. ", "bold"),
                (
                    "Press Enter, and the results in the clipboard will be "
                    "copied automatically"
                ),
            )
        else:
            instructions = Text.assemble(
                ("1. ", "bold"),
                ("Copy the prompt above and paste it into ChatGPT\n"),
                ("2. ", "bold"),
                ("Wait for ChatGPT's complete response\n"),
                ("3. ", "bold"),
                ("Copy ChatGPT's entire response\n"),
                ("4. ", "bold"),
                (
                    "Press Enter, and the results in the clipboard will be "
                    "copied automatically"
                ),
            )

        self.console.print(
            Panel(instructions, title="[cyan]Instructions[/cyan]", border_style="cyan")
        )
        self.console.print()

    def _format_constraints(self, capabilities: dict[str, Any]) -> str:
        """Format capability constraints for display.

        Args:
            capabilities: Capability constraints dictionary

        Returns:
            Formatted constraints string
        """
        constraints = []

        if not capabilities.get("web_access", True):
            constraints.append(
                "🚫 NO WEB BROWSING - Work with provided information only"
            )

        if capabilities.get("json_required", False):
            constraints.append("📋 JSON OUTPUT REQUIRED - Response must be valid JSON")

        if not capabilities.get("tools_allowed", True):
            constraints.append("🔧 NO EXTERNAL TOOLS - Use built-in capabilities only")

        max_retries = capabilities.get("max_retries", 0)
        if max_retries > 0:
            constraints.append(f"🔄 MAX {max_retries} RETRIES allowed if needed")

        return "\n".join(constraints) if constraints else ""

    def _format_constraints_plain(self, capabilities: dict[str, Any]) -> str:
        """Format capability constraints for plain text display.

        Args:
            capabilities: Capability constraints dictionary

        Returns:
            Formatted constraints string without emojis
        """
        constraints = []

        if not capabilities.get("web_access", True):
            constraints.append(
                "* NO WEB BROWSING - Work with provided information only"
            )

        if capabilities.get("json_required", False):
            constraints.append("* JSON OUTPUT REQUIRED - Response must be valid JSON")

        if not capabilities.get("tools_allowed", True):
            constraints.append("* NO EXTERNAL TOOLS - Use built-in capabilities only")

        max_retries = capabilities.get("max_retries", 0)
        if max_retries > 0:
            constraints.append(f"* MAX {max_retries} RETRIES allowed if needed")

        return "\n".join(constraints) if constraints else ""

    def _combine_prompts_for_chatgpt(self, system: str, user: str) -> str:
        """Combine system and user prompts into single prompt for ChatGPT.

        Args:
            system: System prompt with constraints and context
            user: User prompt with the evaluation task

        Returns:
            Combined prompt suitable for ChatGPT
        """
        parts = []

        if system.strip():
            parts.append(f"Context and Instructions:\n{system.strip()}")

        if user.strip():
            parts.append(f"Task:\n{user.strip()}")

        return "\n\n".join(parts)

    def _read_from_clipboard(self) -> str | None:
        """Read text from macOS clipboard using pbpaste.

        Returns:
            Clipboard text if successful, None otherwise
        """
        try:
            process = subprocess.run(
                ["pbpaste"], text=True, check=True, capture_output=True
            )
            return process.stdout if process.returncode == 0 else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to macOS clipboard using pbcopy.

        Args:
            text: Text to copy to clipboard

        Returns:
            True if successful, False otherwise
        """
        try:
            process = subprocess.run(
                ["pbcopy"], input=text, text=True, check=True, capture_output=True
            )
            return process.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _collect_response(self) -> str:
        """Collect manual response from user with timeout handling.

        Returns:
            Raw response text from user

        Raises:
            ProviderTimeoutError: If timeout is reached
            ProviderError: If collection fails
        """

        # Set up timeout handler
        def timeout_handler(signum: int, frame: Any) -> None:
            raise ProviderTimeoutError(
                f"No response received within {self.timeout} seconds"
            )

        # Configure timeout if specified
        if self.timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

        try:
            # Wait for user to press Enter, then check clipboard
            self.console.print(
                "[bold]Press Enter when you have copied ChatGPT's response "
                "to the clipboard:[/bold]"
            )
            input()  # Wait for user to press Enter

            # Now try to read from clipboard
            clipboard_content = self._read_from_clipboard()

            if clipboard_content and clipboard_content.strip():
                # Show preview of clipboard content
                preview = clipboard_content[:500] + (
                    "..." if len(clipboard_content) > 500 else ""
                )
                self.console.print(
                    Panel(
                        Syntax(preview, "text", theme="monokai", word_wrap=True),
                        title="[green]Clipboard Content Found[/green]",
                        border_style="green",
                    )
                )

                use_clipboard = Prompt.ask(
                    "\n[bold]Use this clipboard content as response?[/bold]",
                    choices=["y", "n"],
                    default="y",
                )

                if use_clipboard.lower() == "y":
                    response = clipboard_content.strip()
                else:
                    response = self._collect_manual_input()
            else:
                self.console.print(
                    "[yellow]No content found in clipboard. "
                    "Please paste response manually.[/yellow]"
                )
                response = self._collect_manual_input()

            return response

        finally:
            # Cancel timeout
            if self.timeout > 0:
                signal.alarm(0)

    def _collect_manual_input(self) -> str:
        """Collect manual response input from user.

        Returns:
            Raw response text from user
        """
        self.console.print(
            "[bold]Paste ChatGPT's response below (press Enter twice to submit):[/bold]"
        )

        lines = []
        empty_line_count = 0

        while True:
            try:
                line = input()
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    lines.append(line)
                else:
                    empty_line_count = 0
                    lines.append(line)
            except EOFError:
                break

        response = "\n".join(lines).strip()

        # Confirm response with user
        if response:
            self.console.print()
            self.console.print(
                Panel(
                    Syntax(
                        response[:500] + ("..." if len(response) > 500 else ""),
                        "text",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    title="[green]Response Preview[/green]",
                    border_style="green",
                )
            )

            confirm = Prompt.ask(
                "\n[bold]Use this response?[/bold]", choices=["y", "n"], default="y"
            )

            if confirm.lower() != "y":
                self.console.print(
                    "[yellow]Response discarded. Please try again.[/yellow]"
                )
                return self._collect_manual_input()

        return response
