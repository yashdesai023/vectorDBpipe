
import argparse
import sys
import json
import os
from pathlib import Path
def get_pipeline(config_path):
    from vectorDBpipe.pipeline.text_pipeline import TextPipeline
    return TextPipeline(config_path=config_path)
from vectorDBpipe.config.config_manager import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="VectorDBPipe CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ingest_parser.add_argument("--source", default=None, help="Path to data source (file or directory)")
    
    # Query Command
    query_parser = subparsers.add_parser("query", help="Query the database")
    query_parser.add_argument("text", nargs="?", help="Query text (optional if interactive)") # Make text optional
    query_parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")
    query_parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode for faster queries")

    # Pipeline Run Command (Wrapper for ingest for now)
    pipeline_parser = subparsers.add_parser("pipeline", help="Manage pipelines")
    pipeline_parser.add_argument("action", choices=["run", "list", "stop"], help="Action")
    pipeline_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    if args.command == "ingest":
        try:
            print(f"Initializing pipeline with config: {args.config}")
            pipeline = get_pipeline(args.config)
            
            # Override data_path if source provided
            if args.source:
                # Ensure it's a Path object for the loader
                source_path = Path(args.source)
                pipeline.data_path = source_path
                pipeline.loader.data_path = source_path
                print(f"Overriding data source to: {args.source}")
                
            pipeline.process()
        except Exception as e:
            print(f"Error during ingestion: {e}")
            sys.exit(1)

    elif args.command == "query":
        try:
            if args.interactive:
                print("\n\033[90m" + "─"*60 + "\033[0m")
                print("\033[1;37m VDB INTERACTIVE CHAT\033[0m")
                print("\033[90m Type 'exit' or 'quit' to end session.\033[0m")
                print("\033[90m" + "─"*60 + "\033[0m\n")
                
                print("\033[90mLoading models & database... \033[0m", end="", flush=True)
                pipeline = get_pipeline(args.config)
                print("\033[32mReady\033[0m\n")
                
                while True:
                    try:
                        user_input = input("\033[1;36mYou\n> \033[0m").strip()
                        if user_input.lower() in ['exit', 'quit', 'q']:
                            print("\n\033[90mSession ended.\033[0m")
                            break
                        if not user_input:
                            continue
                            
                        results = pipeline.search(user_input, top_k=args.top_k)
                        
                        print(f"\n\033[1;35mAssistant\033[0m")
                        print("\033[90m" + "─" * 60 + "\033[0m")
                        
                        if not results:
                            print("\033[90mNo relevant context found.\033[0m")
                        else:
                            if pipeline.llm:
                                print("\033[90mThinking...\033[0m", end="\r", flush=True)
                                # Build context
                                context_strs = []
                                for r in results:
                                    ctx = r.get('document', '')
                                    if not ctx:
                                        ctx = r.get('metadata', {}).get('text', '')
                                    context_strs.append(ctx)
                                context_str = "\n\n---\n\n".join(context_strs)
                                
                                response = pipeline.llm.generate_response(
                                    system_prompt="You are an intelligent documentation assistant. Answer the user queries completely by extracting information from the provided context. If the context does not contain the answer, say you do not know based on the context.",
                                    user_query=user_input,
                                    retrieved_context=context_str
                                )
                                print(" "*20, end="\r") # clear the thinking line
                                print(f"\033[37m{response}\033[0m\n")
                                print("\033[90mSources:\033[0m")
                            
                            for i, res in enumerate(results):
                                score = res.get('score', 0)
                                meta = res.get('metadata', {})
                                text = meta.get('text', 'N/A')[:150].replace('\n', ' ')
                                source = meta.get('source', 'Unknown')
                                
                                if not pipeline.llm:
                                    print(f"\033[37m{text} ...\033[0m")
                                print(f"\033[90m[{i+1}] Source: {source} │ Score: {score:.4f}\033[0m")
                            print()
                            
                    except KeyboardInterrupt:
                        print("\n\033[90mSession ended.\033[0m")
                        break
                    except Exception as e:
                        print(f"\033[31mError: {e}\033[0m")

            elif args.text:
                # One-shot query
                pipeline = get_pipeline(args.config)
                results = pipeline.search(args.text, top_k=args.top_k)
                
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print(f"\nSearch Results for: '{args.text}'\n")
                    if not results:
                        print("No results found.")
                    for i, res in enumerate(results):
                        score = res.get('score', 0)
                        meta = res.get('metadata', {})
                        text = meta.get('text', 'N/A')[:200].replace('\n', ' ')
                        source = meta.get('source', 'Unknown')
                        print(f"{i+1}. [{score:.4f}] {source}")
                        print(f"   \"{text}...\"\n")
            else:
                print("Error: query text is required unless using --interactive")
                sys.exit(1)

        except Exception as e:
            print(f"Error during query: {e}")
            sys.exit(1)

    elif args.command == "pipeline":
        if args.action == "run":
             # For now, pipeline run is just ingest
            print(f"Running pipeline (ingest) with config: {args.config}")
            pipeline = get_pipeline(args.config)
            pipeline.process()
        elif args.action == "list":
            print("Active Pipelines: [Default]")
        elif args.action == "stop":
            print("No active pipelines to stop.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
