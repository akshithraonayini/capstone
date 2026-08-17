from src.mcp_client import call_mcp


def main():

    print("\n========== MCP TEST ==========\n")

    result = call_mcp(
        "read_knowledge_document",
        {
            "filename": "wfh_policy.txt"
        }
    )

    print("MCP Result:")
    print(result)


if __name__ == "__main__":
    main()