"""Manual Copilot provider adapter for paste-based evaluation.

This module implements a manual provider that displays prompts and collects
responses via copy-paste, useful when no API is available.
"""

import signal
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from bench.adapters.base import Provider, ProviderError, ProviderTimeoutError


class CopilotManualProvider(Provider):
    """Manual Copilot provider for paste-based interaction.

    Displays prompts to the user and collects responses via manual paste,
    storing raw text for evaluation when API access is not available.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize manual Copilot provider.

        Args:
            **kwargs: Configuration options for manual interaction
        """
        super().__init__(name="Copilot (Manual)", **kwargs)
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

        # Display header
        header = Text("Microsoft Copilot Manual Evaluation", style="bold blue")
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

        # Display system prompt
        if system.strip():
            system_panel = Panel(
                Syntax(system, "text", theme="monokai", word_wrap=True),
                title="[green]System Prompt[/green]",
                border_style="green",
            )
            self.console.print(system_panel)
            self.console.print()

        # Display user prompt
        user_panel = Panel(
            Syntax(user, "text", theme="monokai", word_wrap=True),
            title="[yellow]User Prompt[/yellow]",
            border_style="yellow",
        )
        self.console.print(user_panel)
        self.console.print()

        # Display instructions
        instructions = Text.assemble(
            ("1. ", "bold"),
            ("Copy the prompts above and paste them into Microsoft Copilot\n"),
            ("2. ", "bold"),
            ("Wait for Copilot's complete response\n"),
            ("3. ", "bold"),
            ("Copy Copilot's entire response and paste it below\n"),
            ("4. ", "bold"),
            ("Press Enter twice to submit (or Ctrl+C to cancel)"),
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
            # Collect multi-line response
            self.console.print(
                "[bold]Paste Copilot's response below "
                "(press Enter twice to submit):[/bold]"
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
                    return self._collect_response()

            return response

        finally:
            # Cancel timeout
            if self.timeout > 0:
                signal.alarm(0)
