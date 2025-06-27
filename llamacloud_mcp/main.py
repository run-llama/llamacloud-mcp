import click
import os

from mcp.server.fastmcp import Context, FastMCP
from llama_cloud_services import LlamaExtract
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from typing import Awaitable, Callable, Optional


mcp = FastMCP("llama-index-server")


def make_index_tool(
    index_name: str, project_id: Optional[str], org_id: Optional[str]
) -> Callable[[Context, str], Awaitable[str]]:
    async def tool(ctx: Context, query: str) -> str:
        try:
            await ctx.info(f"Querying index: {index_name} with query: {query}")
            index = LlamaCloudIndex(
                name=index_name,
                project_id=project_id,
                organization_id=org_id,
            )
            response = index.as_query_engine().query(query)
            return str(response)
        except Exception as e:
            await ctx.error(f"Error querying index: {str(e)}")
            return f"Error querying index: {str(e)}"

    return tool


def make_extract_tool(
    agent_name: str, project_id: Optional[str], org_id: Optional[str]
) -> Callable[[Context, str], Awaitable[str]]:
    async def tool(ctx: Context, file_path: str) -> str:
        """Extract data using a LlamaExtract Agent from the given file."""
        try:
            await ctx.info(
                f"Extracting data using agent: {agent_name} with file path: {file_path}"
            )
            llama_extract = LlamaExtract(
                organization_id=org_id,
                project_id=project_id,
            )
            extract_agent = llama_extract.get_agent(name=agent_name)
            result = extract_agent.extract(file_path)
            return str(result)
        except Exception as e:
            await ctx.error(f"Error extracting data: {str(e)}")
            return f"Error extracting data: {str(e)}"

    return tool


@click.command()
@click.option(
    "--index",
    "indexes",
    multiple=True,
    required=False,
    type=str,
    help="Index definition in the format name:description. Can be used multiple times.",
)
@click.option(
    "--extract-agent",
    "extract_agents",
    multiple=True,
    required=False,
    type=str,
    help="Extract agent definition in the format name:description. Can be used multiple times.",
)
@click.option(
    "--project-id", required=False, type=str, help="Project ID for LlamaCloud"
)
@click.option(
    "--org-id", required=False, type=str, help="Organization ID for LlamaCloud"
)
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help='Transport to run the MCP server on. One of "stdio", "sse", "streamable-http".',
)
@click.option("--api-key", required=False, type=str, help="API key for LlamaCloud")
def main(
    indexes: Optional[list[str]],
    extract_agents: Optional[list[str]],
    project_id: Optional[str],
    org_id: Optional[str],
    transport: str,
    api_key: Optional[str],
) -> None:
    api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise click.BadParameter(
            "API key not found. Please provide an API key or set the LLAMA_CLOUD_API_KEY environment variable."
        )
    else:
        os.environ["LLAMA_CLOUD_API_KEY"] = api_key

    # Parse indexes into (name, description) tuples
    index_info = []
    if indexes:
        for idx in indexes:
            if ":" not in idx:
                raise click.BadParameter(
                    f"Index '{idx}' must be in the format name:description"
                )
            name, description = idx.split(":", 1)
            index_info.append((name, description))

    # Parse extract agents into (name, description) tuples if provided
    extract_agent_info = []
    if extract_agents:
        for agent in extract_agents:
            if ":" not in agent:
                raise click.BadParameter(
                    f"Extract agent '{agent}' must be in the format name:description"
                )
            name, description = agent.split(":", 1)
            extract_agent_info.append((name, description))

    # Dynamically register a tool for each index
    for name, description in index_info:
        tool_func = make_index_tool(name, project_id, org_id)
        mcp.tool(name=f"query_{name}", description=description)(tool_func)

    # Dynamically register a tool for each extract agent, if any
    for name, description in extract_agent_info:
        tool_func = make_extract_tool(name, project_id, org_id)
        mcp.tool(name=f"extract_{name}", description=description)(tool_func)

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
