from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_cloud_services import LlamaParse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from llama_index.core.schema import TextNode
from llama_index.core.llms import LLM
from llama_index.core.async_utils import run_jobs, asyncio_run
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from collections import defaultdict
from llama_cloud_services import LlamaExtract
from llama_cloud.core.api_error import ApiError
from llama_cloud import ExtractConfig
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
import pandas as pd
from llama_cloud_services.extract import SourceText
from llama_index.tools.mcp.utils import workflow_as_mcp
from dotenv import load_dotenv
import logging

load_dotenv()


## SETUP
project_id = "2fef999e-1073-40e6-aeb3-1f3c0e64d99b"
organization_id = "43b88c8f-e488-46f6-9013-698e3d2e374a"
# set LLM, embedding model
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
llm = OpenAI(model="gpt-4.1")
Settings.llm = llm
Settings.embed_model = embed_model


split_category_prompt = """\
You are an AI document assistant tasked with finding the 'split categories' given a user description and the document text.
- The split categories is a list of string tags from the document that correspond to the user description.
- Do not make up split categories. 
- Do not include category tags that don't fit the user description,\
for instance subcategories or extraneous titles.
- Do not exclude category tags that do fit the user description. 

For instance, if the user asks to "find all top-level sections of an ArXiv paper", then a sample output would be:
["1. Introduction", "2. Related Work", "3. Methodology", "4. Experiments", "5. Conclusion"]

The split description and document text are given below. 

Split description:
{split_description}

Here is the document text:
{document_text}
    
"""


class SplitCategories(BaseModel):
    """A list of all split categories from a document."""

    split_categories: List[str]


async def afind_split_categories(
    split_description: str,
    nodes: List[TextNode],
    llm: Optional[LLM] = None,
    page_limit: Optional[int] = 5,
) -> List[str]:
    """Find split categories given a user description and the page limit.
    
    These categories will then be used to find the exact splits of the document. 
    
    NOTE: with the page limit there is an assumption that the categories are found in the first few pages,\
    for instance in the table of contents. This does not account for the case where the categories are \
    found throughout the document. 
    
    """
    llm = llm or OpenAI(model="gpt-4.1")

    chat_template = ChatPromptTemplate(
        [
            ChatMessage.from_str(split_category_prompt, "user"),
        ]
    )
    nodes_head = nodes[:page_limit] if page_limit is not None else nodes
    doc_text = "\n-----\n".join(
        [n.get_content(metadata_mode="all") for n in nodes_head]
    )

    result = await llm.astructured_predict(
        SplitCategories,
        chat_template,
        split_description=split_description,
        document_text=doc_text,
    )
    return result.split_categories

# result = await parser.aparse("./data/asset_manager_fund_analysis/fidelity_fund.pdf")
# markdown_nodes = await result.aget_markdown_nodes(split_by_page=True)

split_prompt = """\
You are an AI document assistant tasked with extracting out splits from a document text according to a certain set of rules. 

You are given a chunk of the document text at a time. 
You are responsible for determining if the chunk of the document text corresponds to the beginning of a split. 

We've listed general rules below, and the user has also provided their own rules to find a split. Please extract
out the splits according to the defined schema. 

General Rules: 
- You should ONLY extract out a split if the document text contains the beginning of a split.
- If the document text contains the beginning of two or more splits (e.g. there are multiple sections on a single page), then \
return all splits in the output.
- If the text does not correspond to the beginning of any split, then return a blank list. 
- A valid split must be clearly delineated in the document text according to the user rules. \
Do NOT identify a split if it is mentioned, but is not actually the start of a split in the document text.
- If you do find one or more splits, please output the split_name according to the format \"{split_key}_X\", \
where X is a short tag corresponding to the split. 

Split key:
{split_key}

User-defined rules:
{split_rules}


Here is the chunk text:
{chunk_text}
    
"""


class SplitOutput(BaseModel):
    """The metadata for a given split start given a chunk."""

    split_name: str = Field(
        ..., description="The name of the split (in the format \{split_key\}_X)"
    )
    split_description: str = Field(
        ..., description="A short description corresponding to the split."
    )
    page_number: int = Field(..., description="Page number of the split.")


class SplitsOutput(BaseModel):
    """A list of all splits given a chunk."""

    splits: List[SplitOutput]


async def atag_splits_in_node(
    split_rules: str, split_key: str, node: TextNode, llm: Optional[LLM] = None
):
    """Tag split in a single node."""
    llm = llm or OpenAI(model="gpt-4")

    chat_template = ChatPromptTemplate(
        [
            ChatMessage.from_str(split_prompt, "user"),
        ]
    )

    result = await llm.astructured_predict(
        SplitsOutput,
        chat_template,
        split_rules=split_rules,
        split_key=split_key,
        chunk_text=node.get_content(metadata_mode="all"),
    )
    return result.splits


async def afind_splits(
    split_rules: str, split_key: str, nodes: List[TextNode], llm: Optional[LLM] = None
) -> Dict:
    """Find splits."""

    # tag each node with split or no-split
    tasks = [atag_splits_in_node(split_rules, split_key, n, llm=llm) for n in nodes]
    async_results = await run_jobs(tasks, workers=8, show_progress=True)
    all_splits = [s for r in async_results for s in r]

    split_name_to_pages = defaultdict(list)

    split_idx = 0
    for idx, n in enumerate(nodes):
        cur_page = n.metadata["page_number"]

        # update the current split if needed
        while (
            split_idx + 1 < len(all_splits)
            and all_splits[split_idx + 1].page_number <= cur_page
        ):
            split_idx += 1

        # add page number to the current split
        if all_splits[split_idx].page_number <= cur_page:
            split_name = all_splits[split_idx].split_name
            split_name_to_pages[split_name].append(cur_page)

    return split_name_to_pages


# put it all together - detect categories, then split document based on those categories
async def afind_categories_and_splits(
    split_description: str,
    split_key: str,
    nodes: List[TextNode],
    additional_split_rules: Optional[str] = None,
    llm: Optional[LLM] = None,
    page_limit: int = 5,
    verbose: bool = False,
):
    """Find categories and then splits."""
    categories = await afind_split_categories(
        split_description, nodes, llm=llm, page_limit=page_limit
    )
    if verbose:
        logging.info(f"Split categories: {categories}")
    full_split_rules = f"""Please split by these categories: {categories}"""
    if additional_split_rules:
        full_split_rules += f"\n\n\n{additional_split_rules}"

    return await afind_splits(full_split_rules, split_key, nodes, llm=llm)



# Define output schema
class FundData(BaseModel):
    """Concise fund data extraction schema optimized for LLM extraction"""

    # Identifiers
    fund_name: str = Field(
        ...,
        description="Full fund name exactly as it appears, e.g. 'Fidelity Asset Manager® 20%'",
    )
    target_equity_pct: Optional[int] = Field(
        None,
        description="Target equity percentage from fund name (20, 30, 40, 50, 60, 70, or 85)",
    )
    report_date: Optional[str] = Field(
        None, description="Report date in YYYY-MM-DD format, e.g. '2024-09-30'"
    )

    # Asset Allocation (as percentages, e.g. 27.4 for 27.4%)
    equity_pct: Optional[float] = Field(
        None,
        description="Actual equity allocation percentage from 'Equity Central Funds' section",
    )
    fixed_income_pct: Optional[float] = Field(
        None,
        description="Fixed income allocation percentage from 'Fixed-Income Central Funds' section",
    )
    money_market_pct: Optional[float] = Field(
        None,
        description="Money market allocation percentage from 'Money Market Central Funds' section",
    )
    other_pct: Optional[float] = Field(
        None,
        description="Other investments percentage (Treasury + Investment Companies + other)",
    )

    # Primary Share Class Metrics (use the main retail class, usually named after the fund)
    nav: Optional[float] = Field(
        None,
        description="Net Asset Value per share for the main retail class (e.g. Asset Manager 20% class)",
    )
    net_assets_usd: Optional[float] = Field(
        None,
        description="Total net assets in USD for the main retail class from 'Net Asset Value' section",
    )
    expense_ratio: Optional[float] = Field(
        None,
        description="Expense ratio as percentage (e.g. 0.48 for 0.48%) from Financial Highlights",
    )
    management_fee: Optional[float] = Field(
        None,
        description="Management fee rate as percentage from Financial Highlights or Notes",
    )

    # Performance (as percentages)
    one_year_return: Optional[float] = Field(
        None,
        description="One-year total return percentage from Financial Highlights (e.g. 13.74 for 13.74%)",
    )
    portfolio_turnover: Optional[float] = Field(
        None, description="Portfolio turnover rate percentage from Financial Highlights"
    )

    # Risk Metrics (in USD)
    equity_futures_notional: Optional[float] = Field(
        None,
        description="Net notional amount of equity futures contracts (positive if net long, negative if net short)",
    )
    bond_futures_notional: Optional[float] = Field(
        None,
        description="Net notional amount of bond/treasury futures contracts (positive if net long, negative if net short)",
    )

    # Fund Flows (in USD)
    net_investment_income: Optional[float] = Field(
        None,
        description="Net investment income for the period from Statement of Operations",
    )
    total_distributions: Optional[float] = Field(
        None,
        description="Total distributions to shareholders from Statement of Changes in Net Assets",
    )
    net_asset_change: Optional[float] = Field(
        None,
        description="Net change in assets from beginning to end of period (end minus beginning net assets)",
    )


class FundComparisonData(BaseModel):
    """Flattened data optimized for CSV export and analysis"""

    funds: list[FundData]

    def to_csv_rows(self) -> list[dict]:
        """Convert to list of dictionaries for CSV export"""
        return [fund.dict() for fund in self.funds]




async def aextract_data_over_split(
    extract_agent: LlamaExtract,
    split_name: str,
    page_numbers: List[int],
    nodes: List[TextNode],
    llm: Optional[LLM] = None,
) -> FundData:
    """Extract fund data for a given split."""

    # combine node text that matches the page numbers
    filtered_nodes = [n for n in nodes if n.metadata["page_number"] in page_numbers]
    filtered_text = "\n-------\n".join(
        [n.get_content(metadata_mode="all") for n in filtered_nodes]
    )
    result_dict = (
        await extract_agent.aextract(SourceText(text_content=filtered_text))
    ).data

    fund_data = FundData.model_validate(result_dict)

    return fund_data


async def aextract_data_over_splits(
    extract_agent: LlamaExtract,
    split_name_to_pages: Dict[str, List],
    nodes: List[TextNode],
    llm: Optional[LLM] = None,
):
    """Extract fund data for each split, aggregate."""
    tasks = [
        aextract_data_over_split(extract_agent, split_name, page_numbers, nodes, llm=llm)
        for split_name, page_numbers in split_name_to_pages.items()
    ]
    all_fund_data = await run_jobs(tasks, workers=8, show_progress=True)
    return FundComparisonData(funds=all_fund_data)



class TypedStartEvent(StartEvent):
    """Typed start event."""
    file_path: str

class ParseDocEvent(Event):
    nodes: List[TextNode]


class DocSplitEvent(Event):
    split_name_to_pages: Dict[str, List[int]]
    nodes: List[TextNode]


class FidelityFundExtraction(Workflow):
    """
    Workflow to extract data from a solar panel datasheet and generate a comparison report
    against provided design requirements.
    """

    def __init__(
        self,
        parser: LlamaParse,
        extract_agent: LlamaExtract,
        split_description: str = "Find and split by the main funds in this document, should be listed in the first few pages",
        split_rules: str = """
        - You must split by the name of the fund
        - Each fund will have a list of tables underneath it, like schedule of investments, financial statements
        - Each fund usually has schedule of investments right underneath it 
        - Do not tag the cover page/table of contents
        """,
        split_key: str = "fidelity_asset_manager",
        llm: Optional[LLM] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.parser = parser
        self.extract_agent = extract_agent
        self.split_description = split_description
        self.split_rules = split_rules
        self.split_key = split_key
        self.llm = llm

    @step
    async def parse_doc(self, ctx: Context, ev: TypedStartEvent) -> ParseDocEvent:
        """Parse document into markdown nodes."""
        result = await self.parser.aparse(file_path=ev.file_path)
        markdown_nodes = await result.aget_markdown_nodes(split_by_page=True)
        return ParseDocEvent(nodes=markdown_nodes)

    @step
    async def find_splits(self, ctx: Context, ev: ParseDocEvent) -> DocSplitEvent:
        split_name_to_pages = await afind_categories_and_splits(
            self.split_description,
            self.split_key,
            ev.nodes,
            additional_split_rules=self.split_rules,
            llm=self.llm,
            verbose=True,
        )
        return DocSplitEvent(
            split_name_to_pages=split_name_to_pages,
            nodes=ev.nodes,
        )

    @step
    async def run_extraction(self, ctx: Context, ev: DocSplitEvent) -> StopEvent:
        all_fund_data = await aextract_data_over_splits(
            self.extract_agent, ev.split_name_to_pages, ev.nodes, llm=self.llm
        )
        all_fund_data_df = pd.DataFrame(all_fund_data.to_csv_rows())
        return StopEvent(
            result={
                "all_fund_data": all_fund_data,
                "all_fund_data_df": all_fund_data_df,
            }
        )




## SETUP THE WORKFLOW

parser = LlamaParse(
    premium_mode=True,
    result_type="markdown",
    project_id=project_id,
    organization_id=organization_id,
)

llama_extract = LlamaExtract(
    show_progress=True,
    check_interval=5,
    project_id=project_id,
    organization_id=organization_id,
)
extract_config = ExtractConfig(extraction_mode="BALANCED")



try:
    existing_agent = llama_extract.get_agent(name="FundDataExtractor2")
    if existing_agent:
        # # Deletion can take some time since all underlying files will be purged
        # llama_extract.delete_agent(existing_agent.id)
        extract_agent = existing_agent
    else:
        extract_agent = llama_extract.create_agent(
            "FundDataExtractor2", data_schema=FundData, config=extract_config
        )
         
except ApiError as e:
    if e.status_code == 404:
        pass
    else:
        raise

# Create and run workflow
workflow = FidelityFundExtraction(
    parser=parser, extract_agent=extract_agent, verbose=True, timeout=None
)


mcp = workflow_as_mcp(workflow)

if __name__ == "__main__":
    logging.info("running mcp server")
    mcp.run(transport="stdio")

