from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
import os
import argparse
from mcp.server.fastmcp.resources import TextResource
from llama_cloud_services import LlamaExtract
from mcp.server.fastmcp import Context

load_dotenv()

mcp = FastMCP('llama-index-server')

@mcp.tool()
async def llama_index_documentation(query: str) -> str:
    """Search the llama-index documentation for the given query."""
    index_name_resource = await mcp._resource_manager.get_resource("resource://index")
    project_name_resource = await mcp._resource_manager.get_resource("resource://project")
    org_id_resource = await mcp._resource_manager.get_resource("resource://org")

    index_name = await index_name_resource.read()
    project_name = await project_name_resource.read()
    org_id = await org_id_resource.read()
    
    index = LlamaCloudIndex(
        name=index_name,
        project_name=project_name,
        organization_id=org_id,
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    )

    response = index.as_query_engine().query(query + " Be verbose and include code examples.")
    return str(response)

@mcp.tool()
async def extract_data(file_path: str) -> str:
    """Extract data using a LlamaExtract Agent from the given file. 
    The file_path will be provided by the MCP client and should be the full file path accessible with the filesystem server"""
    try:
        print("Got file path: "+ file_path)
        llama_extract = LlamaExtract()
        extract_agent = llama_extract.get_agent(name="invoice-extractor")
        result = extract_agent.extract(file_path)
        return str(result)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LlamaCloud MCP server with custom configuration')
    parser.add_argument('--index-name', type=str, required=True,
                      help='Name of the LlamaCloud index')
    parser.add_argument('--project-name', type=str, default="Default",
                      help='Name of the LlamaCloud project')
    parser.add_argument('--org-id', type=str, required=True,
                      help='Organization ID for LlamaCloud')
    
    args = parser.parse_args()

    index_resource = TextResource(
        uri="resource://index",
        name="LlamaIndex Name",
        text=args.index_name,
    )
    project_resource = TextResource(
        uri="resource://project",
        name="LlamaCloud Project Name",
        text=args.project_name,
    )
    org_resource = TextResource(
        uri="resource://org",
        name="LlamaCloud Organization ID",
        text=args.org_id,
    )
        
    mcp.add_resource(index_resource)
    mcp.add_resource(project_resource)
    mcp.add_resource(org_resource)
    
    mcp.run(transport="stdio")
