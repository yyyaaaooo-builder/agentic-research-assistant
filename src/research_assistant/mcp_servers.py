"""MCP servers the agent can connect to."""

from research_assistant.tools.mcp_client import McpServerConfig

MCP_SERVERS: dict[str, McpServerConfig] = {
    "arxiv": McpServerConfig(
        name="arxiv",
        command="uvx",
        args=("arxiv-mcp-server@0.8.2",),
        tools=("download_paper", "get_paper_outline", "read_paper_section"),
        # Pin every argument that controls output size, so long raw text never enters the context.
        pinned_args={
            "download_paper": {"max_chars": 1, "return_full_text": False},
            "read_paper_section": {"return_full_text": False},
        },
        descriptions={
            "download_paper": (
                "Download an arXiv paper so its outline and sections can be read. "
                "Returns a short confirmation, not the paper's text."
            ),
        },
    )
}
