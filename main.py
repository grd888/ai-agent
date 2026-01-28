import sys
from typing import Any
from agent.agent import Agent
import asyncio
import click

from agent.events import AgentEventType
from ui.tui import TUI, get_console

console = get_console()

class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI(console)

    async def run_single(self, message: str) -> str | None:
        async with Agent() as agent:
            self.agent = agent
            return await self.__process_message(message)

    async def __process_message(self, message: str) -> str | None:
        if not self.agent:
            return None
        
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                self.tui.stream_assistant_delta(content)


@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None):
    cli = CLI()
    # messages = [{"role": "user", "content": prompt}]
    print(f"Prompt: {prompt}")
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        print(f"Result: {result}")
        if result is None:
            sys.exit(1)
            


if __name__ == "__main__":
    main()
