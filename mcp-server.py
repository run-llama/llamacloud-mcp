from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
import os
import click
from mcp.server.fastmcp.resources import TextResource
from llama_cloud_services import LlamaExtract
from mcp.server.fastmcp import Context

load_dotenv()

mcp = FastMCP('llama-index-server')

def make_index_tool(index_name, project_name, org_id):
    async def tool(query: str) -> str:
        index = LlamaCloudIndex(
            name=index_name,
            project_name=project_name,
            organization_id=org_id,
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        )
        response = index.as_query_engine().query(query + " Be verbose and include code examples.")
        return str(response)
    return tool

def make_extract_tool(agent_name):
    async def tool(file_path: str) -> str:
        """Extract data using a LlamaExtract Agent from the given file."""
        try:
            print(f"Got file path: {file_path} for agent: {agent_name}")
            llama_extract = LlamaExtract()
            extract_agent = llama_extract.get_agent(name=agent_name)
            result = extract_agent.extract(file_path)
            return str(result)
        except Exception as e:
            return f"Error extracting data: {str(e)}"
    return tool

@click.command()
@click.option('--index', 'indexes', multiple=True, required=True, type=str, help='Index definition in the format name:description. Can be used multiple times.')
@click.option('--extract-agent', 'extract_agents', multiple=True, required=False, type=str, help='Extract agent definition in the format name:description. Can be used multiple times.')
@click.option('--project-name', default="Default", type=str, help='Name of the LlamaCloud project')
@click.option('--org-id', required=True, type=str, help='Organization ID for LlamaCloud')
def main(indexes, extract_agents, project_name, org_id):
    # Parse indexes into (name, description) tuples
    index_resources = []
    index_info = []
    for idx in indexes:
        if ':' not in idx:
            raise click.BadParameter(f"Index '{idx}' must be in the format name:description")
        name, description = idx.split(':', 1)
        index_resources.append(TextResource(
            uri=f"resource://index/{name}",
            name=f"Index Name: {name}",
            text=name,
        ))
        index_resources.append(TextResource(
            uri=f"resource://index/{name}/description",
            name=f"Index Description: {name}",
            text=description,
        ))
        index_info.append((name, description))
    # Parse extract agents into (name, description) tuples if provided
    extract_agent_info = []
    if extract_agents:
        for agent in extract_agents:
            if ':' not in agent:
                raise click.BadParameter(f"Extract agent '{agent}' must be in the format name:description")
            name, description = agent.split(':', 1)
            extract_agent_info.append((name, description))
    project_resource = TextResource(
        uri="resource://project",
        name="LlamaCloud Project Name",
        text=project_name,
    )
    org_resource = TextResource(
        uri="resource://org",
        name="LlamaCloud Organization ID",
        text=org_id,
    )
    for res in index_resources:
        mcp.add_resource(res)
    mcp.add_resource(project_resource)
    mcp.add_resource(org_resource)

    # Dynamically register a tool for each index
    for name, description in index_info:
        tool_func = make_index_tool(name, project_name, org_id)
        mcp.tool(name=f"query_{name}", description=description)(tool_func)

    # Dynamically register a tool for each extract agent, if any
    for name, description in extract_agent_info:
        tool_func = make_extract_tool(name)
        mcp.tool(name=f"extract_{name}", description=description)(tool_func)

    mcp.run(transport="stdio")

if __name__ == "__main__":
   main()
