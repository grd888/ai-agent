import sys
from agent.agent import Agent
import asyncio
import click
from pathlib import Path
from agent.events import AgentEventType
from config.config import Config
from config.loader import load_config
from ui.tui import TUI, get_console

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.agent: Agent | None = None
        self.tui = TUI(config, console)
        self.config = config

    async def run_single(self, message: str) -> str | None:
        async with Agent(self.config) as agent:
            self.agent = agent
            return await self.__process_message(message)

    async def run_interactive(self) -> str | None:
        self.tui.print_welcome(
            "AI Agent",
            lines=[
                f"model: {self.config.model_name}",
                f"cwd: {self.config.cwd}",
                "commands: /help /config /approval /model /exit",
            ],
        )
        async with Agent(self.config) as agent:
            self.agent = agent

            while True:
                try:
                    user_input = console.input("\n[user]>[/user] ").strip()
                    if not user_input:
                        continue
                    await self.__process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to quit[/dim]")
                except EOFError:
                    console.print("\n")
                    break

        console.print("\n[dim]Goodbye![/dim]")

    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool_kind = None
        tool = self.agent.tool_registry.get(tool_name)
        if tool:
            tool_kind = tool.kind.value

        return tool_kind

    async def __process_message(self, message: str) -> str | None:
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None

        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)

            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming = False

            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "An error occurred")
                console.print(f"\n[error]Error: {error}[/error]")

            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                tool = self.agent.tool_registry.get(tool_name)
                if tool:
                    tool_kind = tool.kind.value
                self.tui.tool_call_start(
                    event.data.get("call_id"),
                    tool_name,
                    tool_kind,
                    event.data.get(
                        "arguments",
                        {},
                    ),
                )

            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                call_id = event.data.get("call_id")
                self.tui.tool_call_complete(
                    call_id,
                    tool_name,
                    tool_kind,
                    event.data.get("success", False),
                    event.data.get("output", ""),
                    event.data.get("error", None),
                    event.data.get("metadata", None),
                    event.data.get("truncated", False),
                )
        return final_response


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Working directory for the agent",
)
def main(prompt: str | None, cwd: Path | None):

    try:
        config = load_config(cwd=cwd)
    except Exception as e:
        console.print(f"[error]Error loading config: {e}[/error]")
        sys.exit(1)

    errors = config.validate()
    if errors:
        console.print("[error]Configuration errors:[/error]")
        for error in errors:
            console.print(f"[error]{error}[/error]")
        sys.exit(1)

    cli = CLI(config)
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
